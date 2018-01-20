import numpy
import matplotlib.pyplot as plt
import cv2
import time

#nombre d'images de cube
n = 3
#les images de base
base = []

#les solutions des images de base
sol = []


#setup de larray stockage des coordonnees
cd = [[[0,0,0,0]]]

ind = int(0)
cl_ind = int(0)

##fonction dessin de carres + stockage de coords
def coord(event,x,y,param,i) :
    global ind, cl_ind
    if event == cv2.EVENT_LBUTTONDOWN:
        ##stockage des coordonnees
        cd[i][cl_ind][ind] = x
        cd[i][cl_ind][ind+1] = y
        ind = ind +2
        if(ind == 4) :
            ##affichage du rectangle
            cv2.rectangle(sol[i],(cd[i][cl_ind][0],cd[i][cl_ind][1]),(cd[i][cl_ind][2],cd[i][cl_ind][3]),(0,0,255),1,8,0)
            ##update de limage
            cv2.imshow('Window'+str(i),sol[i])
            ##update des arrays 
            ind = 0
            cd[i].append([0,0,0,0])
            cl_ind = cl_ind + 1
            
for i in range(0,n) :
    ##update de larray base + sol
    base.append(cv2.imread('cube'+str(i+1)+'.jpg',1))
    base[i] = cv2.resize(base[i],(320,240), interpolation = cv2.INTER_CUBIC)
    sol.append(base[i])
    ##update de larray cd
    cd.append([[0,0,0,0]])
    ##affichage de limage
    cv2.imshow('Window'+str(i),base[i])
    ##appel fonction dessin de carres + stockage de coords 
    cv2.setMouseCallback('Window'+str(i),coord,i)
    ok = 0
    while(1 == 1) :
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    cl_ind = 0

##cv2.waitKey(0)
##cv2.destroyAllWindows()

##actuellement, le code ne stock quune seule version des images : celle avec les carres rouges. Il faut aussi avoir une version sans les carres. (cest vraiment facile a faire mais la flemme)
