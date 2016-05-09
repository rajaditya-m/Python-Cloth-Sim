from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *
import sys, os, traceback
if sys.platform == 'win32' or sys.platform == 'win64':
    os.environ['SDL_VIDEO_CENTERED'] = '1'
from math import *
import numpy as np
#import random

from scipy import sparse
#from scipy.sparse.linalg import spsolve
from numpy.linalg import solve, norm
from numpy.random import rand


pygame.display.init()
pygame.font.init()

screen_size = [1280,720]
multisample = 16
icon = pygame.Surface((1,1)); icon.set_alpha(0); pygame.display.set_icon(icon)
pygame.display.set_caption("Simple Demo")
if multisample:
    pygame.display.gl_set_attribute(GL_MULTISAMPLEBUFFERS,1)
    pygame.display.gl_set_attribute(GL_MULTISAMPLESAMPLES,multisample)
pygame.display.set_mode(screen_size,OPENGL|DOUBLEBUF)

#glEnable(GL_BLEND)
#glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)

#glEnable(GL_TEXTURE_2D)
#glTexEnvi(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,GL_MODULATE)
#glTexEnvi(GL_POINT_SPRITE,GL_COORD_REPLACE,GL_TRUE)

glHint(GL_PERSPECTIVE_CORRECTION_HINT,GL_NICEST)
glEnable(GL_DEPTH_TEST)

glPointSize(4)

def subtract(vec1,vec2):
    return [vec1[i]-vec2[i] for i in [0,1,2]]

def add(vec1,vec2):
     return [vec1[i]+vec2[i] for i in [0,1,2]]

def get_length(vec):
    return sum([vec[i]*vec[i] for i in [0,1,2]])**0.5

def objreader(filename):
    f = open(filename,'r')
    vertList = []
    #vertNormalList = []
    #uvList = []

    triList = []
    triTextList = []
    triNormalList = []

    for line in f:
        linestrp = line.strip()
        lineelem = [i for i in linestrp.split(' ')]
        if(lineelem[0]=='v'):
            x = float(lineelem[1])
            y = float(lineelem[2])
            z = float(lineelem[3])
            vertList.append([x,y,z])
        elif(lineelem[0]=='f'):
            eltri = []
            eltritext = []
            eltrinorm = []
            for j in range(1,len(lineelem)):
                el = lineelem[j].split('/')
                if(len(el)==1):
                    eltri.append(int(el[0])-1)
                elif(len(el)==3):
                    eltri.append(int(el[0])-1)
                    if(el[1]!=''):
                        eltritext.append(int(el[1])-1)
                    eltrinorm.append(int(el[2])-1)
            triList.append(eltri)
            triTextList.append(eltritext)
            triNormalList.append(eltrinorm)

    numpy_vertList = np.array(vertList)
    numpy_triList = np.array(triList)
    numpy_triTextList = np.array(triTextList)
    numpy_triNormalList = np.array(triNormalList)

    return [numpy_vertList,numpy_triList,numpy_triTextList,numpy_triNormalList]

def edgeinformation(triList):
    #Lets make a set DS class 
    edge_dict = dict()
    for t in range(0,len(triList)):
        a = triList[t][0]
        b = triList[t][1]
        c = triList[t][2]
        
        if(a<b):
            e1 = [a,b]
        else:
            e2 = [b,a]

        if(a<c):
            e2 = [a,c]
        else:
            e2 = [c,a]

        if(b<c):
            e3 = [b,c]
        else:
            e3 = [c,b]

        if tuple(e1) in edge_dict:
            pair_list = edge_dict[tuple(e1)]
            edge_dict[tuple(e1)] = [pair_list[0],t]
        else:
            edge_dict[tuple(e1)] = [t,-1]

        if tuple(e2) in edge_dict:
            pair_list = edge_dict[tuple(e2)]
            edge_dict[tuple(e2)] = [pair_list[0],t]
        else:
            edge_dict[tuple(e2)] = [t,-1]

        if tuple(e3) in edge_dict:
            pair_list = edge_dict[tuple(e3)]
            edge_dict[tuple(e3)] = [pair_list[0],t]
        else:
            edge_dict[tuple(e3)] = [t,-1]

    return edge_dict    

def generateSprings(edgeDict,triList,vertList):
    stretchSprings = []
    restLenStretchSprings = []
    bendingSprings = []
    restLenBendingSprings = []

    for key in edgeDict:
        stretchSprings.append([key[0],key[1]])
        v1 = vertList[key[0]]
        v2 = vertList[key[1]]
        restLen = np.linalg.norm(v1-v2)
        restLenStretchSprings.append(restLen)

        #now we will check if the spring is a bending spring candidate 
        value = edgeDict[key]
        if(value[0]!=-1 and value[1]!=-1):

            l1 = triList[value[0]]
            l2 = triList[value[1]]

            l1 = np.delete(l1,np.where(l1==key[0]))
            l1 = np.delete(l1,np.where(l1==key[1]))

            l2 = np.delete(l2,np.where(l2==key[0]))
            l2 = np.delete(l2,np.where(l2==key[1]))

            bendingSprings.append([l1[0],l2[0]])
            v1 = vertList[l1[0]]
            v2 = vertList[l2[0]]
            restLen  = np.linalg.norm(v1-v2)

            restLenBendingSprings.append(restLen)

    return [stretchSprings,restLenStretchSprings,bendingSprings,restLenBendingSprings]



def add3x3MatrixBlockToCOOSparseMatrixVector(vector,tupletable,mat,a,b):
    for i in range(0,3):
        for j in range(0,3):
            r = a+i
            c = b+j
            loc = tupletable[tuple([a,b])]
            vector[loc] = vector[loc] + mat(a,b)
            





class MassSpringCloth(object):
    def __init__(self,filename):

        [self.vertList,self.triList,triTextList,triNormalList] = objreader(filename)

        self.edgeDict = edgeinformation(self.triList)

        [self.stretchSprings,self.restLenStretchSprings,self.bendingSprings,self.restLenBendingSprings] = generateSprings(self.edgeDict,self.triList,self.vertList)

        self.numTris = len(self.triList)

        self.numVerts = len(self.vertList)

        self.n3 = self.numVerts*3

        velList = [[0,0,0]]*self.numVerts

        self.velocity = np.array(velList)

        self.defineCOOSparsityStructures()

        


    #def addStretchSpringForcesAndJacobians():


    def defineCOOSparsityStructures(self):
        K_I = []
        K_J = []
        self.stiffness_Matrix_Dict = dict()
        counter = 0

        #first initialize the diagonal 3x3 matrices 
        for i in range(0,self.numVerts):
            for j in range(0,3):
                for k in range(0,3):
                    K_I.append(3*i+j)
                    K_J.append(3*i+k)
                    self.stiffness_Matrix_Dict[tuple([3*i+j,3*i+k])] = counter
                    counter += 1

        #Next initialize the stretch 3x3 matrices
        for i in self.stretchSprings:
            v1 = i[0]
            v2 = i[1]
            for j in range(0,3):
                for k in range(0,3):
                    K_I.append(3*v1+j)
                    K_J.append(3*v2+k)
                    self.stiffness_Matrix_Dict[tuple([3*v1+j,3*v2+k])] = counter
                    counter += 1

            for j in range(0,3):
                for k in range(0,3):
                    K_I.append(3*v2+j)
                    K_J.append(3*v1+k)
                    self.stiffness_Matrix_Dict[tuple([3*v2+j,3*v1+k])] = counter
                    counter += 1

        #Next initialize the bending 3x3 matrices
        for i in self.bendingSprings:
            v1 = i[0]
            v2 = i[1]
            for j in range(0,3):
                for k in range(0,3):
                    K_I.append(3*v1+j)
                    K_J.append(3*v2+k)
                    self.stiffness_Matrix_Dict[tuple([3*v1+j,3*v2+k])] = counter
                    counter += 1

            for j in range(0,3):
                for k in range(0,3):
                    K_I.append(3*v2+j)
                    K_J.append(3*v1+k)
                    self.stiffness_Matrix_Dict[tuple([3*v2+j,3*v1+k])] = counter
                    counter += 1

        self.stiffnessMatrix_I = np.array(K_I)
        self.stiffnessMatrix_J = np.array(K_J)



    def addSpringForcesAndJacobians(self):
        k = 100.0
        kd = 0.001
        kb = 0.1
        cb = k
        I = np.identity(3)
        h = 0.033

        for idx in range(0,3):#len(bendingSprings)):
            i = self.bendingSprings[idx][0]
            j = self.bendingSprings[idx][1]
            r = self.restLenBendingSprings[idx]

            xi = self.vertList[i]
            xj = self.vertList[j]

            vi = self.velocity[i]
            vj = self.velocity[j]

            xij = xi - xj
            vij = vi - vj

            xij_norm = np.linalg.norm(xij)
            xij_hat = xij/xij_norm

            if(1):#(xij_norm>r):
                f_i = (-k)*(xij_norm-r)*xij_hat
                f_j = -f_i

                outerPdk = np.outer(xij_hat,xij_hat)
                J_Fi_xi = outerPdk + (1.0-(r/xij_norm))*(I - outerPdk)
                J_Fi_xi = J_Fi_xi * (-k)
                J_Fj_xi = -J_Fi_xi

                df_dx_i = h*(J_Fi_xi * vi + J_Fj_xi * vj)
                df_dx_j = h*(J_Fj_xi * vi + J_Fi_xi * vj)














    def draw_wireframe(self):

        #points in red
        glColor3f(1.0,0.0,0.0)
        glBegin(GL_POINTS)
        for particle in self.vertList:
            glVertex3fv(particle)
        glEnd()

        glColor3f(0.0,0.0,1.0)
        glBegin(GL_LINES)
        for x in self.stretchSprings:
            glVertex3fv(self.vertList[x[0]])
            glVertex3fv(self.vertList[x[1]])
        glEnd()

        glColor3f(0.0,1.0,0.0)
        glBegin(GL_LINES)
        for x in self.bendingSprings:
            glVertex3fv(self.vertList[x[0]])
            glVertex3fv(self.vertList[x[1]])
        glEnd()

    def draw_mesh(self):
        glBegin(GL_TRIANGLES)
        for x in self.triList:
            glVertex3fv(self.vertList[x[0]])
            glVertex3fv(self.vertList[x[1]])
            glVertex3fv(self.vertList[x[2]])
        glEnd()

    def update(self):
        #DUMMY = 0

        self.addSpringForcesAndJacobians()

        #stiffness_Matrix_V = [0]*self.n3
        stiffness_Matrix_V = rand(len(self.stiffnessMatrix_I))
        A = sparse.coo_matrix((stiffness_Matrix_V,(self.stiffnessMatrix_I,self.stiffnessMatrix_J)),shape=(self.n3,self.n3))

        for i in range(0,len(self.vertList)):
            np.add(self.vertList[i],np.array([0.0,-0.1,0.0]),self.vertList[i])

       



class Particle(object):
    def __init__(self,pos):
        self.pos = pos
        self.last_pos = list(self.pos)
        self.accel = [0.0,0.0,0.0]
        
        self.constrained = False
    def move(self,dt):
        #Don't move constrained particles
        if self.constrained: return
        #Move
        for i in [0,1,2]:
            #Verlet
            temp = 2*self.pos[i] - self.last_pos[i] + self.accel[i]*dt*dt
            self.last_pos[i] = self.pos[i]
            self.pos[i] = temp
            
            if self.pos[i] < 0.0: self.pos[i] = 0.0
            elif self.pos[i] > 1.0: self.pos[i] = 1.0
    def draw(self):
        glVertex3fv(self.pos)

class Edge(object):
    def __init__(self, p1,p2, tolerance=0.1):
        self.p1 = p1
        self.p2 = p2

        self.tolerance = tolerance
        
        self.rest_length = get_length(subtract(self.p2.pos,self.p1.pos))
        self.lower_length = self.rest_length*(1.0-self.tolerance)
        self.upper_length = self.rest_length*(1.0+self.tolerance)
    def constrain(self):
        vec = [self.p2.pos[i]-self.p1.pos[i] for i in [0,1,2]]
        length = get_length(vec)

        if   length > self.upper_length:
            target_length = self.upper_length
            strength = 1.0
        elif length < self.lower_length:
            target_length = self.lower_length
            strength = 1.0
        elif length > self.rest_length:
            target_length = self.rest_length
            strength = (length - self.rest_length) / ( 0.1*self.rest_length)
        elif length < self.rest_length:
            target_length = self.rest_length
            strength = (length - self.rest_length) / (-0.1*self.rest_length)
        else:
            return
        movement_for_each = strength * (length - target_length) / 2.0

        for i in [0,1,2]:
            if not self.p1.constrained: self.p1.pos[i] += movement_for_each*vec[i]
            if not self.p2.constrained: self.p2.pos[i] -= movement_for_each*vec[i]

class ClothCPU(object):
    def __init__(self, res):
        self.res = res

        corners = [
            [-1,-1],        [ 1,-1],

            [-1, 1],        [ 1, 1]
        ]
        edges = [
                    [ 0,-1],
            [-1, 0],        [ 1, 0],
                    [ 0, 1]
        ]
        self.rels = edges + corners
        for rel in self.rels:
            length = sum([rel[i]*rel[i] for i in [0,1]])**0.5
            rel.append(length/float(self.res))

        self.reset()
    def reset(self):
        self.particles = []
        for z in range(self.res):
            row = []
            for x in range(self.res):
                row.append(Particle([
                    float(x)/float(self.res-1),
                    1.0,
                    float(z)/float(self.res-1)
                ]))
            self.particles.append(row)
        self.particles[         0][         0].constrained = True
        self.particles[self.res-1][         0].constrained = True
        self.particles[         0][self.res-1].constrained = True

        self.edges = []
        for z1 in range(self.res):
            for x1 in range(self.res):
                p1 = self.particles[z1][x1]
                for rel in self.rels:
                    x2 = x1 + rel[0]
                    z2 = z1 + rel[1]
                    if x2 < 0 or x2 >= self.res: continue
                    if z2 < 0 or z2 >= self.res: continue
                    p2 = self.particles[z2][x2]

                    found = False
                    for edge in self.edges:
                        if edge.p1 == p2:
                            found = True
                            break
                    if found: continue

                    self.edges.append(Edge(p1,p2))

    def constrain(self, n):
        for constraint_pass in range(n):
            for edge in self.edges:
                edge.constrain()
    def update(self,dt):
        #Gravity
        for row in self.particles:
            for particle in row:
                particle.accel = [0.0, gravity, 0.0]
        #Move everything
        for row in self.particles:
            for particle in row:
                particle.move(dt)

    def draw(self):
        glBegin(GL_POINTS)
        for row in self.particles:
            for particle in row:
                particle.draw()
        glEnd()
    def draw_wireframe(self):
        glBegin(GL_LINES)
        for edge in self.edges:
            glVertex3fv(edge.p1.pos)
            glVertex3fv(edge.p2.pos)
        glEnd()

    def draw_mesh(self):
        for z in range(self.res-1):
            glBegin(GL_QUAD_STRIP)
            for x in range(self.res):
                glVertex3fv(self.particles[z  ][x].pos)
                glVertex3fv(self.particles[z+1][x].pos)
            glEnd()

gravity = -9.8
cloth = ClothCPU(30)
mspcloth = MassSpringCloth('flat_cloth.obj')

camera_rot = [-90,-40]#[70,23]
camera_radius = 14.5 #2.5
camera_center = [0,0,0]#[0.5,0.5,0.5]

#these are ancilliary functions that we are writing for our help 


def get_input():
    global camera_rot, camera_radius
    keys_pressed = pygame.key.get_pressed()
    mouse_buttons = pygame.mouse.get_pressed()
    mouse_rel = pygame.mouse.get_rel()
    for event in pygame.event.get():
        if   event.type == QUIT: return False
        elif event.type == KEYDOWN:
            if   event.key == K_ESCAPE: return False
            elif event.key == K_r: cloth.reset()
        elif event.type == MOUSEBUTTONDOWN:
            if   event.button == 4: camera_radius -= 0.5
            elif event.button == 5: camera_radius += 0.5
    if mouse_buttons[0]:
        camera_rot[0] += mouse_rel[0]
        camera_rot[1] += mouse_rel[1]
    return True
def update(dt):
    mspcloth.update()
    #cloth.constrain(3)
    #cloth.update(dt)

def draw():
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    
    glViewport(0,0,screen_size[0],screen_size[1])
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45,float(screen_size[0])/float(screen_size[1]), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    camera_pos = [
        camera_center[0] + camera_radius*cos(radians(camera_rot[0]))*cos(radians(camera_rot[1])),
        camera_center[1] + camera_radius                            *sin(radians(camera_rot[1])),
        camera_center[2] + camera_radius*sin(radians(camera_rot[0]))*cos(radians(camera_rot[1]))
    ]
    gluLookAt(
        camera_pos[0],camera_pos[1],camera_pos[2],
        camera_center[0],camera_center[1],camera_center[2],
        0,1,0
    )

    #cloth.draw_mesh()

    glColor3f(1,1,1)
    #cloth.draw_wireframe()

    #mspcloth.draw_mesh()
    mspcloth.draw_wireframe()


    #glColor3f(1,0,0)
    # glBegin(GL_LINES)
    # points = []
    # for x in [0,1]:
    #     for y in [0,1]:
    #         for z in [0,1]:
    #             points.append([x,y,z])
    # for p1 in points:
    #     for p2 in points:
    #         unequal = sum([int(p1[i]!=p2[i]) for i in [0,1,2]])
    #         if unequal == 1:
    #             glVertex3fv(p1)
    #             glVertex3fv(p2)
    # glEnd()

    glColor3f(1,1,1)
    
    pygame.display.flip()

def main():
    target_fps = 60
    clock = pygame.time.Clock()
    while True:
        if not get_input(): break

        dt = 1.0/float(target_fps)
        update(dt)
        
        draw()
        clock.tick(target_fps)
    pygame.quit()
if __name__ == '__main__':
    try:
        main()
    except:
        traceback.print_exc()
        pygame.quit()
        input()
