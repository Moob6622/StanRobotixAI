
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import chainer

class ImageLabelisation (object):

    def __init__ (self, imageNb):
        self.n = imageNb
        self.base = []
        self.solution = []
        self.rectCoords  = [[[0,0,0,0]]]
        self.index = int(0)                #numero de click
        self.cubeIndex = int(0)            #nombre de boite 
        self.bboxDict = {}                 #boites formatees pour chainer
        
    def Coord (self,event,x,y,param,i):

        if event == cv2.EVENT_LBUTTONDOWN:
            ##stockage des coordonnees
            self.rectCoords[i][self.cubeIndex][self.index] = x
            self.rectCoords[i][self.cubeIndex][self.index+1] = y
            
            self.index = self.index +2
            
        if(self.index == 4) :
            ##affichage du rectangle
            cv2.rectangle(self.solution[i],(self.rectCoords[i][self.cubeIndex][0],self.rectCoords[i][self.cubeIndex][1]),(self.rectCoords[i][self.cubeIndex][2] ,self.rectCoords[i][self.cubeIndex][3]),(0,0,255),1,8,0)
            
            ##update de limage
            cv2.imshow('Window'+str(i),self.solution[i])
            
            ##update des arrays et du dico
            
            self.bboxDict['cube'+str(i+1)] = np.append (self.bboxDict['cube'+str(i+1)], [self.rectCoords[i][self.cubeIndex]], axis = 0)
            #print(str(self.bboxDict['cube'+str(i+1)+'.jpg']) + '\n')
            
            
            self.index = 0
            self.rectCoords[i].append([0,0,0,0])
            self.cubeIndex  = self.cubeIndex + 1

    def DrawRect (self):
        for i in range(0,self.n) :
            print (i)
            ##update de larray base + sol
            self.base.append(cv2.imread('./imgs/cube'+str(i+1)+'.jpg',1))
            self.base[i] = cv2.resize(self.base[i],(320,240), interpolation = cv2.INTER_CUBIC)
            self.solution.append(self.base[i])
            
            ##update de l'array coord
            self.rectCoords.append([[0,0,0,0]])
            
            self.bboxDict['cube'+str(i+1)] = np.zeros((0, 4))
            
            ##affichage de limage
            cv2.imshow('Window'+str(i),self.base[i])
            
            ##appel fonction dessin de carres + stockage de coords 
            cv2.setMouseCallback('Window'+str(i),self.Coord,i)
            
            while(True) :
                if cv2.waitKey(20) & 0xFF == 27:
                    break
            cv2.destroyAllWindows()
            self.cubeIndex = 0
        return self.bboxDict

        
            
if __name__ == '__main__' :
    import ImageLabelization

    x = ImageLabelization.ImageLabelisation(3)
    y = x.DrawRect()
    np.savez('./data/bboxs' , *list(y.values()), **y)
    z = np.load('./data/bboxs.npz')
    z.files


