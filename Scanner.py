import cv2
import numpy as np
from scipy.interpolate import griddata
from collections import deque
import test_keras_model as model

#colors
red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)
black = (0,0,0)
white = (255,255,255)
#read Image
def readImage(name):
    img = cv2.imread("./"+name+".jpg")
    img = cv2.resize(img, (640,480))
    return img

def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(img, (5,5), 0)
    thresh_blur = cv2.adaptiveThreshold(blur_img, 255, 1, 1, 19, 5)
    return thresh_blur

def findSquare(Modified_Sudoku_Image):
    Temp_Image = Modified_Sudoku_Image.copy()
    #cv2.imshow('Temp_Image', Temp_Image)
    ###Create a copy of the Modified_Sudoku_Image to implement cv2.findContours() on,
    ## since after applying this method the image gets distorted for some reason.
    ###We are using the .copy() method to create the image since using something like
    ## img1 = img2, simply creates an object pointing to the original one. So altering
    ## either of the images also alters the other image and hence using it makes no sense
    Contours, Hierarchy = cv2.findContours(Temp_Image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #OGimg = cv2.cvtColor(Temp_Image,cv2.COLOR_GRAY2RGB)
    #cv2.drawContours(OGimg,Contours,-1,(0,255,0),1)
    #cv2.imshow("Modified_Sudoku_Image.png", OGimg)
    ###Find the contours in the image
    ###cv2.findContours(image, mode, method[, contours[, hierarchy[, offset]]])
    ###(image) ~input binary image
    ###Refer the link below for more info
    ##http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#findcontours
    Required_Square_Contour = None
    Required_Contour_Area = 0
    for Contour in Contours:
        Contour_Area = cv2.contourArea(Contour)
        ###Calculates the area enclosed by the vector of 2D points denoted by
        ## the variable Contour
        if Contour_Area > 500:
            if Contour_Area > Required_Contour_Area:
                Required_Contour_Area = Contour_Area
                Required_Square_Contour = Contour
    ###Code for finding out the largest contour (on the basis of area)
    Perimeter_of_Contour = cv2.arcLength(Required_Square_Contour, True)
    ###Calculates a contour perimeter or a curve length
    ###cv2.arcLength(curve, closed)
    ###(curve) ~Input vector of 2D points
    ###(closed) ~Flag indicating whether the curve is closed or not
    Temp_Square_Countour = cv2.approxPolyDP(Required_Square_Contour, 0.05*Perimeter_of_Contour, True)
    ###Approximates a polygonal curve(s) with the specified precision
    ###cv2.approxPolyDP(curve, epsilon, closed[, approxCurve])
    Temp_Square_Countour = Temp_Square_Countour.tolist()
    Approx_Square_Countour = []
    for Temp_Var_1 in Temp_Square_Countour:
        for Temp_Var_2 in Temp_Var_1:
            Temp_Var_2[0], Temp_Var_2[1] = Temp_Var_2[1], Temp_Var_2[0]
            Approx_Square_Countour.append(Temp_Var_2)
    ###Temp_Square_Countour has the coordinates inside a list within a list,
    ## hence to extract it we're doing this. Also we're changing (row, column) i.e.
    ## (y, x) to (column, row) i.e. (x, y)
    ###This was done because the griddata function from the scipy library
    ## takes in values as (column, row) i.e. (x,y) instead of (row, column) i.e (y,x)
    Approx_Square_Countour = deque(Approx_Square_Countour)
    ###Applying deque function on anything converts it into a queue and we can use
    ## functions like popleft() etc on it, as if it were a queue
    Min_Sum = 9999999
    ###Initialized to a fairly large number as we want minimum
    Counter = -1
    ###Used as counter to keep tract of the iteration number so that the
    ## location of top-left coordinate can be stored in the variable Loc
    Loc = 0
    for i in Approx_Square_Countour:
        Counter+=1
        if Min_Sum > sum(i):
            Min_Sum = sum(i)
            Loc = Counter
    if Loc != 0:
        for i in range(0,Loc):
             Approx_Square_Countour.append(Approx_Square_Countour[0])
             Approx_Square_Countour.popleft()
    ###If the sum of the x and y coordinates is minimum it would automatically
    ## mean that the coordinate refers to the top-left point of the square.
    ###We know the coordinates of the square are stored in a cyclic fashion,
    ## hence if we know the location of the top-left coordinate then we can
    ## re-arrage it by appending the 1st coordinate and then poping it.
    ## Example: (4,1,2,3)
    ## Now appending 1st loc we get (4,1,2,3,4)
    ## Now popping 1st loc we get (1,2,3,4) which is the required result
    ## That is what this code does to rearrange the coordinates
    Approx_Square_Countour[1], Approx_Square_Countour[3] = Approx_Square_Countour[3], Approx_Square_Countour[1]
    ###Flipping the location of 1st and 3rd coordinates makes the coordinate
    ## pointer go counter-clockwise. We do this because opencv stores the
    ## coordinate values in a clockwise fashion, however griddata function from
    ## scipy library requires it to be in a counter-clockwise fashion
    #cv2.drawContours(Modified_Sudoku_Image,[Approx_Square_Countour],0,255,10)
    Mask = np.zeros((Modified_Sudoku_Image.shape),np.uint8)
    ###Creates a black image of the same size as the input image
    cv2.drawContours(Mask,[Required_Square_Contour],0,255,-1)
    cv2.drawContours(Mask,[Required_Square_Contour],0,0,2)
    ###Overwrites the black image with the area of the sudoku in white
    Modified_Sudoku_Image = cv2.bitwise_and(Modified_Sudoku_Image,Mask)
    ###Compares the Modified_Sudoku_Image and the Mask and blackens all parts
    ## of the image other than the sudoku
    #cv2.imshow('Modified_Sudoku_Image', Modified_Sudoku_Image)
    return Modified_Sudoku_Image, Approx_Square_Countour


def stretchSquare(Modified_Sudoku_Image, Square_Contour):
    grid_x, grid_y = np.mgrid[0:449:450j, 0:449:450j]
    ###Creates grid_x such that it is a 2D array having all values equal to their corresponding row value
    ## Creates grid_y such that it is a 2D array having all values equal to their corresponding column value
    destination = np.array([[0,0],[0,449],[449,449],[449,0]])
    ###Denotes the coordinates of the corners of the destination onto which we want to map the sudoku
    source = np.asarray(Square_Contour)
    ###Denotes the coordinates of the corners of the sudoku as present in the source
    grid_z = griddata(destination, source, (grid_x, grid_y), method='cubic')
    map_x = np.append([], [ar[:,1] for ar in grid_z]).reshape(450,450)
    map_y = np.append([], [ar[:,0] for ar in grid_z]).reshape(450,450)
    ###Converts the values to stretch/contract the image and accordingly adjust pixel values
    map_x_32 = map_x.astype('float32')
    map_y_32 = map_y.astype('float32')
    ###Converts the values to the specified type
    Warped_Sudoku_Image = cv2.remap(Modified_Sudoku_Image, map_x_32, map_y_32, cv2.INTER_CUBIC)
    ###Used to remap the sudoku from the original image into a new image of
    ## size 450x450, this size was chosen because identifying each small block becomes easier
    ## since it'll have a size of 50x50
    Warped_Sudoku_Image = cv2.bitwise_not(Warped_Sudoku_Image)
    ###Inverts the color scheme
    #cv2.imshow("Warped_Sudoku_Image.png", Warped_Sudoku_Image)
    cv2.imwrite("Temp_Storage/Warped_Sudoku_Image.png", Warped_Sudoku_Image)
    return Warped_Sudoku_Image

def convertGrid(Warped_Sudoku_Image):
    ROI_X_Width = 28
    ROI_Y_Height = 28

    Modified_Sudoku_Image = cv2.adaptiveThreshold(Warped_Sudoku_Image, 255, 1, 1, 11, 5)
    Contours,Hierarchy = cv2.findContours(Modified_Sudoku_Image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    Sudoku_Grid = [[0] * 9 for i in range(9)]
    for Contour in Contours:
        if cv2.contourArea(Contour)>30 and cv2.contourArea(Contour)<1000:
            [Abscissa,Ordinate,X_Width,Y_Height] = cv2.boundingRect(Contour)
            A, O, X, Y = Abscissa, Ordinate, X_Width, Y_Height
            if  Y_Height>20 and Y_Height<35:
                Abscissa -= 9
                Ordinate -= 5
                X_Width += 15
                Y_Height += 8
                cv2.rectangle(Warped_Sudoku_Image,(Abscissa,Ordinate),(Abscissa+X_Width,Ordinate+Y_Height),(0,255,0),2)
                Region_of_Interest = Modified_Sudoku_Image[Ordinate:Ordinate+Y_Height,Abscissa:Abscissa+X_Width]
                Region_of_Interest = cv2.resize(Region_of_Interest,(ROI_X_Width,ROI_Y_Height))
                Region_of_Interest = Region_of_Interest.reshape((1,ROI_X_Width*ROI_Y_Height))
                Region_of_Interest = np.float32(Region_of_Interest)
                result = model.pred(Region_of_Interest)
                Sudoku_Grid[(O + Y) // 50][A // 50] = result
    return Sudoku_Grid


def flowChart(image_name):
    img = readImage(image_name)
    img = preProcessing(img)
    Mod, cnt = findSquare(img)
    Wimg = stretchSquare(Mod, cnt)
    x = convertGrid(Wimg)
    cv2.imshow("y", Wimg)
    cv2.waitKey(5000)
    return x