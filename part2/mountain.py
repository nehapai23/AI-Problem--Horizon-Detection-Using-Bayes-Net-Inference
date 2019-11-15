#!/usr/local/bin/python3
#
# Authors: nibafna-nrpai-sjejurka
#          Nikita Bafna- Neha Pai- Shivali Jejurkar
#
# Mountain ridge finder
# Based on skeleton code by D. Crandall, Oct 2019
#

from PIL import Image
from numpy import *
from scipy.ndimage import filters
import sys
import imageio
import numpy as np

cols = 0
rows = 0
# calculate "Edge strength map" of an image
#
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale,0,filtered_y)
    return sqrt(filtered_y**2)

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_edge(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( int(max(y-int(thickness/2), 0)), int(min(y+int(thickness/2), image.size[1]-1 )) ):
            image.putpixel((x, t), color)
    return image

#Return the best ridge row numbers for question 1
def ridge_simple(edge_strength):
    ridgeRow = []
    for col in range(0, cols):
        s = [edge_strength[row][col] for row in range(0, rows)]
        ridgeRow.append(s.index(max(s)))
    return ridgeRow
    
#Calculate Emission probablities
def emission(index,col):
    #return col[index]/max(col)
    #return col[index]**0.90
    return col[index]/sum(col)

#Calculate Transmission probablities
def transmission(a,b):
    (r1,c1),(r2,c2) = a,b
    #return sqrt((r2-r1)**2+(c2-c1)**2)/rows
    #return (rows - abs(r2-r1))**2
    if abs(r2-r1) == 0:
        return 1
    else:    
        return 1/abs(r2-r1)

#Utility Log function
def log(x):
    if x <= 0 or x == -inf or x == inf:
        return math.inf
    else:
        return -math.log(x)

#Viterbi function
def viterbi(obs, edge_strength):
    by_cols = np.transpose(edge_strength)
    #Calculate for initial state t=1
    prob = {}
    f = open("demofile.txt", "a")
    for r in range(0,rows):     
        prob[(0,r)] = (log((edge_strength[r][0]/sum(by_cols[0])) * emission(r,by_cols[0])), -1)
        #prob[(0,r)] = (log(1/rows * emission(obs[0],by_cols[0])), -1)
    #Calculate for t>1
    #iterating for all columns
    for c in range(1,len(obs)):
        print(c)
        for r in range(0,rows):
            #emission_prob = emission(obs[c],by_cols[c])
            emission_prob = emission(r,by_cols[c])
            #We are using log probablities so we want to minimize negative log probablities to maximize probablity
            min_val = math.inf
            min_row = -1
            for rr in range(0,rows):
                tr = transmission((rr,c-1),(r,c))
                prob_j = prob[(c-1,rr)][0]+log(tr)
                if (prob_j < min_val):
                    min_val = prob_j
                    min_row = rr
            #Calculating vj(t+1)        
            prob[(c,r)] = (min_val+log(emission_prob), min_row)
    val = []
    l = []
    f.write(str(prob))
    #Backtrackking from last to first based on max values
    for r in range(0,rows):
        val.append(prob[(cols-1,r)][0])
        l.append(prob[(cols-1,r)][1])
    s = l[val.index(min(val))]
    val = [s]
    for c in range(cols-1,0,-1):
        val += [prob[(c,s)][1]]
        s = prob[(c,s)][1]
    f.close()
    return val[::-1]


#Viterbi function
def viterbi_human(obs, edge_strength, row_coord, col_coord):
    by_cols = np.transpose(edge_strength)
    #Calculate for initial given state
    prob = {}
    f = open("demofile.txt", "a")
    for r in range(0,rows):     
        if row_coord == r:
            prob[(col_coord,r)] = (0,-1)
        else:    
            prob[(col_coord,r)] = (log((edge_strength[r][col_coord]/sum(by_cols[col_coord])) * emission(r,by_cols[0])), -1)
    
    #Calculate for remaining states
    #Start from given column coordinate and use viterbi to last column
    for c in range(col_coord+1,cols):
        print(c)
        for r in range(0,rows):
            #emission_prob = emission(obs[c],by_cols[c])
            emission_prob = emission(r,by_cols[c])
            min_val = math.inf
            min_row = -1
            for rr in range(0,rows):
                tr = transmission((rr,c-1),(r,c))
                prob_j = prob[(c-1,rr)][0]+log(tr)
                if (prob_j < min_val):
                    min_val = prob_j
                    min_row = rr
            prob[(c,r)] = (min_val+log(emission_prob), min_row)

    val = []
    l = []
    #Backtrackking
    for r in range(0,rows):
        val.append(prob[(cols-1,r)][0])
        l.append(prob[(cols-1,r)][1])
    s = l[val.index(min(val))]
    val = [s]
    ind = [cols-1]
    for c in range(cols-2,col_coord-1,-1):
        val += [prob[(c,s)][1]]
        ind += [c]
        s = prob[(c,s)][1]
    val = val[::-1]
    ind = ind[::-1]

    #Now perform viterbi from given column to 0 backwards
    for c in range(col_coord-1,-1,-1):
        print(c)
        for r in range(0,rows):
            #emission_prob = emission(obs[c],by_cols[c])
            emission_prob = emission(r,by_cols[c])
            min_val = math.inf
            min_row = -1
            for rr in range(0,rows):
                tr = transmission((rr,c+1),(r,c))
                prob_j = prob[(c+1,rr)][0]+log(tr)
                if (prob_j < min_val):
                    min_val = prob_j
                    min_row = rr
            prob[(c,r)] = (min_val+log(emission_prob), min_row)



    val1 = []
    l = []
    ind = []
    # f.write(str(prob))
    #Backtrackking
    for r in range(0,rows):
        val1.append(prob[(0,r)][0])
        l.append(prob[(0,r)][1])
    s = l[val1.index(min(val1))]
    val1 = [s]
    ind = [0]
    for c in range(1,col_coord,1):
        val1 += [prob[(c,s)][1]]
        s = prob[(c,s)][1]
        ind += [c]
    f.close()
    return val1 + val



# main program
#
(input_filename, gt_row, gt_col) = sys.argv[1:]

# load in image 
input_image = Image.open(input_filename)

# compute edge strength mask
edge_strength = edge_strength(input_image)
cols = len(edge_strength[0])
rows = len(edge_strength)    
imageio.imwrite('edges.jpg', uint8(255 * edge_strength / (amax(edge_strength))))

# You'll need to add code here to figure out the results! For now,
# just create a horizontal centered line.
#ridge = [ edge_strength.shape[0]/2 ] * edge_strength.shape[1]

ridge = ridge_simple(edge_strength)
#Q1 answer
input_image = Image.open(input_filename)
imageio.imwrite("output_simple.jpg", np.array(draw_edge(input_image, ridge, (0, 0, 255), 5)))
print("done")

#Q2 answer
observed = ridge
input_image = Image.open(input_filename)
viterbi_ridge = viterbi(observed, edge_strength)
imageio.imwrite("output_map.jpg", np.array(draw_edge(input_image, viterbi_ridge, (255, 0, 0), 5)))

# Q3 answer
observed = ridge
input_image = Image.open(input_filename)
viterbi_ridge_human = viterbi_human(observed, edge_strength, int(gt_row), int(gt_col))
imageio.imwrite("output_human.jpg", np.array(draw_edge(input_image, viterbi_ridge_human, (0, 255, 0), 5)))
