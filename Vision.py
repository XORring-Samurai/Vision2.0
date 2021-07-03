import vision_arena
import gym
import cv2.aruco as aruco
import cv2
import numpy as np
import time
import os
import sys                                                                                            
import pybullet as p
import pybullet_data
import math

up_x = 25
up_y = 25
down_x = 485
down_y = 485
width = down_x - up_x
height = down_y - up_y
block_width = width / 9
block_height = height / 9

def Transform (lst):
    """
    utility method to transform a list into grid points
    Arguments:
        List of points to be transformed
    Return Value:
        Transformed List
    """
    new_lst = []
    for point in lst:
        x = (point[0] - up_x) // block_width + 1
        y = (point[1] - up_y) // block_height + 1
        x = int(x)
        y = int(y)
        new_lst.append([y, x])
    return new_lst


class Arena:
    def __init__ (self):
        self.env = gym.make("vision_arena-v0")
        self.top = self.env.camera_feed()

        self.rot_speed = 100
        self.trans_speed = 100

        # PID Parameters
        # alter speed such that the bot doesn't fall off due to inertia
        # fast in the beginning, slows down till it reaches the destination
        self.ka = 0.8
        self.kt = 1.0

#--------------------------------------------------------------------------------------------------------------------------
    def Read_aruco (self):
        """
        Method to determine the centre, direction of movement and grid location
        of the bot
        Arguments:
            None
        Return Value:
            a tuple consisting of centre point, direction and grid location
            centre is in (x, y) format
            grid location is in (r, c) format
        """
        img = self.env.camera_feed()
        p.stepSimulation()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        parameters = aruco.DetectorParameters_create()
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters = parameters)

        blank = np.zeros(gray.shape, np.uint8)
        points_x = np.array([corners[0][0][0][0], corners[0][0][1][0], 
                                corners[0][0][2][0], corners[0][0][3][0]], dtype = np.uint64)
        points_y = np.array([corners[0][0][0][1], corners[0][0][1][1],
                                corners[0][0][2][1], corners[0][0][3][1]], dtype = np.uint64)
    
        min_x = np.min(points_x, axis = 0)
        max_x = np.max(points_x, axis = 0)
        min_y = np.min(points_y, axis = 0)
        max_y = np.max(points_y, axis = 0)

        centre_x = (min_x + max_x) / 2
        centre_y = (min_y + max_y) / 2
        centre = (centre_x, centre_y)

        x = (centre_x - up_x) // block_width + 1
        y = (centre_y - up_y) // block_height + 1
        x = int(x)
        y = int(y)
        grid_pos = (y, x)

        direction = (corners[0][0][0][0] - corners[0][0][3][0], corners[0][0][0][1] - corners[0][0][3][1])

        return (centre, direction, grid_pos)

#--------------------------------------------------------------------------------------------------------------------------
    def Distance (self, A, B):
        """
        Method to find Euclidean Distance b/w 2 points.
        Arguments:
            2 tuples b/w which, we need to find the distance.
        Return Value:
            The Distance b/w the given points.
        """
        return math.sqrt( ((A[0] - B[0]) ** 2) + ((A[1] - B[1]) ** 2))


    
    def Angle (self, A, B):
        """
        Method to find angle b/w 2 vectors
        Arguments:
            2 vectors b/w which, the angle needs to be determined.
        Return Value:
            The Angle in degress between the two vectors.
        """
        a = complex(A[0], A[1])
        b = complex(B[0], B[1])
        angle = np.angle(a / b)
        degree = math.degrees(angle)
        return degree

#--------------------------------------------------------------------------------------------------------------------------
    def Rotate (self, fin):
        img = self.env.camera_feed()

        while True:
            (centre, cur, pos) = self.Read_aruco()

            dx = fin[0] - centre[0]
            dy = fin[1] - centre[1]
            angle = self.Angle(cur, (dx, dy))
            # print("Angle = ", angle)

            rot_vel = int(self.rot_speed - self.ka * (90 - abs(angle)))

            if (self.Distance(centre, fin) > 7):
                if abs(angle) < 10:
                    break

                if angle > 0:
                    if angle > 90:
                        for i in range(rot_vel):
                            p.stepSimulation()
                            self.env.move_husky(-3, 3, -3, 3)
                    else:
                        for i in range(rot_vel):
                            p.stepSimulation()
                            self.env.move_husky(-1.8, 3, -1.8, 3)
                else:
                    if angle < -1 * 90:
                        for i in range(rot_vel):
                            p.stepSimulation()
                            self.env.move_husky(3, -1.8, 3, -1.8)
                    else:
                        for i in range(rot_vel):
                            p.stepSimulation()
                            self.env.move_husky(3, -1.8, 3, -1.8)
            else:
                for i in range(self.trans_speed):
                    p.stepSimulation()
                    self.env.move_husky(0, 0, 0, 0)
                break

#------------------------------------------------------------------------------------------------------------------------- 
    def Move (self, B):
        # print(B)
        img = self.env.camera_feed()

        ul = int(B[1] * block_width + up_x)
        ll = int((B[1] - 1) * block_width + up_x)
        ud = int(B[0] * block_height + up_y)
        ld = int((B[0] - 1) * block_height + up_y)
        fin = ((ul + ll) / 2, (ud + ld) / 2)

        # print(fin)

        (centre, cur, pos) = self.Read_aruco()
        # print("BotPos ", centre)

        while True:
            (centre, cur, pos) = self.Read_aruco()
            # print("BotPos ", centre)

            dist = self.Distance(centre, fin)
            trans_vel = int(self.trans_speed - self.kt * (60 - dist))

            if dist < 7:
                break

            self.Rotate(fin)

            for i in range(trans_vel):
                p.stepSimulation()
                self.env.move_husky(3, 3, 3, 3)
    
    #------------------------------------------------------------------------------------------------------------------------- 
    def Extract (self):
        """
        Method to extract colored shapes from the camera feed.
        Arguments: 
            None
        Return value: 
            Tuple of lists of points of all 6 shapes.
            points are in (row, col) format
        """
        rt = []
        rs = []
        rc = []
        yt = []
        ys = []
        yc = []

        img = self.env.camera_feed()

        lower_red = np.array([0, 0, 0], np.uint8)
        upper_red = np.array([80, 80, 255], np.uint8)
        mask_red = cv2.inRange(img, lower_red, upper_red)
        red = cv2.bitwise_and(img, img, mask = mask_red)

        lower_yellow = np.array([0, 40, 40], np.uint8)
        upper_yellow = np.array([40, 255, 255], np.uint8)
        mask_yellow = cv2.inRange(img, lower_yellow, upper_yellow)
        yellow = cv2.bitwise_and(img, img, mask = mask_yellow)

        red_gray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
        yellow_gray = cv2.cvtColor(yellow, cv2.COLOR_BGR2GRAY)

        ret, thresh_red = cv2.threshold(red_gray, 42, 255, cv2.THRESH_BINARY)
        ret, thresh_yellow = cv2.threshold(yellow_gray, 100, 255, cv2.THRESH_BINARY)

        contours_red, hierarchy = cv2.findContours(thresh_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, hierarchy = cv2.findContours(thresh_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        blank_red = np.zeros(red_gray.shape, np.uint8)
        blank_yellow = np.zeros(yellow_gray.shape, np.uint8)

        for contour in contours_red:
            ar = cv2.contourArea(contour)
            if ar > 100:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 3:
                    cv2.drawContours(blank_red, [approx], -1, 255, -1)
                    M = cv2.moments(approx)
                    if M['m00'] != 0:
                        cx = M['m10'] / M['m00']
                        cy = M['m01'] / M['m00']
                        cx = int(cx)
                        cy = int(cy)
                        rt.append([cx, cy])
        
                elif len(approx) == 4:
                    # cv2.drawContours(blank_red, [approx], -1, 150, -1)
                    M = cv2.moments(approx)
                    if M['m00'] != 0:
                        cx = M['m10'] / M['m00']
                        cy = M['m01'] / M['m00']
                        cx = int(cx)
                        cy = int(cy)
                        rs.append([cx, cy])
                
                elif len(approx) > 4:
                    # cv2.drawContours(blank_red, [approx], -1, 200, -1)
                    M = cv2.moments(approx)
                    if M['m00'] != 0:
                        cx = M['m10'] / M['m00']
                        cy = M['m01'] / M['m00']
                        cx = int(cx)
                        cy = int(cy)
                        rc.append([cx, cy])


        for contour in contours_yellow:
            ar = cv2.contourArea(contour)
            if ar > 100: 
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 3:
                    cv2.drawContours(blank_yellow, [approx], -1, 255, -1)
                    M = cv2.moments(approx)
                    if M['m00'] != 0:
                        cx = M['m10'] / M['m00']
                        cy = M['m01'] / M['m00']
                        cx = int(cx)
                        cy = int(cy)
                        yt.append([cx, cy])
                
                elif len(approx) == 4:
                    # cv2.drawContours(blank_yellow, [approx], -1, 150, -1)
                    M = cv2.moments(approx)
                    if M['m00'] != 0:
                        cx = M['m10'] / M['m00']
                        cy = M['m01'] / M['m00']
                        cx = int(cx)
                        cy = int(cy)
                        ys.append([cx, cy])

                elif len(approx) > 4:
                    # cv2.drawContours(blank_yellow, [approx], -1, 200, -1)
                    M = cv2.moments(approx)
                    if M['m00'] != 0:
                        cx = M['m10'] / M['m00']
                        cy = M['m01'] / M['m00']
                        cx = int(cx)
                        cy = int(cy)
                        yc.append([cx, cy])        


        # cv2.imshow('Thresh_red', thresh_red)
        # cv2.imshow('Thresh_yellow', thresh_yellow)
        # cv2.imshow('blank_yellow', blank_yellow)
        # cv2.imshow('blank_red', blank_red)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        rt = Transform(rt)
        rs = Transform(rs)
        rc = Transform(rc)
        yt = Transform(yt)
        ys = Transform(ys)
        yc = Transform(yc)

        return (rt, rs, rc, yt, ys, yc)


#--------------------------------------------------------------------------------------------------------------------------
    class Graph:
        def __init__(self):
            self.matrix = np.zeros((10, 10), np.uint8)
            self.adj = {}
            for i in range(1, 10):
                for j in range(1, 10):
                    self.adj[(i, j)] = []

        def Assign (self, lst, id):
            """
            classifies each grid box into one of the 7 types.
            0 --> Black
            1 --> Red square
            2 --> Red Triangle
            3 --> Red Circle
            4 --> Yellow square
            5 --> Yellow Triangle
            6 --> Yellow Circle
            Argument:
                list of points of a particular type
            Return Val:
                None
            """
            for point in lst:
                r = point[0]
                c = point[1]
                self.matrix[r][c] = id
    
    #----------------------------------------------------------------------------------------------------------------------
        def Build_Main (self):
            """
            Builds adjacency list
            Arguments:
                None
            Return Val:
                None
            """
            for row in range(8, 0, -1):
                self.adj[(row + 1, 1)].append((row, 1))
            
            for col in range(2, 10):
                self.adj[(1, col - 1)].append((1, col))
            
            for row in range(2, 10):
                self.adj[(row - 1, 9)].append((row, 9))
            
            for col in range(8, 0, -1):
                self.adj[(9, col + 1)].append((9, col))
            
            for row in range(6, 2, -1):
                self.adj[(row + 1, 3)].append((row, 3))
            
            for col in range(4, 8):
                self.adj[(3, col - 1)].append((3, col))

            for row in range(4, 8):
                self.adj[(row - 1, 7)].append((row, 7))

            for col in range(6, 2, -1):
                self.adj[(7, col + 1)].append((7, col))

            self.adj[(5, 1)].append((5, 2))
            self.adj[(5, 2)].append((5, 3))
            self.adj[(1, 5)].append((2, 5))
            self.adj[(2, 5)].append((3, 5))
            self.adj[(5, 9)].append((5, 8))
            self.adj[(5, 8)].append((5, 7))
            self.adj[(9, 5)].append((8, 5))
            self.adj[(8, 5)].append((7, 5))
            self.adj[(5, 2)].append((5, 1))
            self.adj[(5, 3)].append((5, 2))
            self.adj[(2, 5)].append((1, 5))
            self.adj[(3, 5)].append((2, 5))
            self.adj[(5, 7)].append((5, 8))
            self.adj[(5, 8)].append((5, 9))
            self.adj[(7, 5)].append((8, 5))
            self.adj[(8, 5)].append((9, 5))
        
    #----------------------------------------------------------------------------------------------------------------------
        def Bfs (self, A, B):
            '''
            Breadth First Search
            Arguments:
                2 Points A and B b/w which we need to find a path
            Return value:
                List of points thru which we can reach B from A.
                if B is not reachable from A, the function returns a list
                of one member containing only A
                else the first element of the returned list is A and the second
                element is B.
            '''
            parent = {}
            for i in range(1, 10):
                for j in range(1, 10):
                    parent[(i, j)] = (-1, -1)

            mark = np.zeros((10, 10), np.uint8)
            path = []

            queue = [A]
            mark[A[0]][A[1]] = 1

            while queue:
                r = queue[0][0]
                c = queue[0][1]
                queue.pop(0)

                if B[0] == r and B[1] == c:
                    break

                for (rr, cc) in self.adj[(r, c)]:
                    if mark[rr][cc] == 0:
                        mark[rr][cc] = 1
                        parent[(rr, cc)] = (r, c)
                        queue.append((rr, cc))

            (r, c) = (B[0], B[1])
            while r != -1:
                path.append((r, c))
                (r, c) = parent[(r, c)]

            path.reverse()

            if path[0][0] != A[0] or path[0][1] != A[1]:
                path.clear()
                path.append(A)

            return path
            
grid = Arena()
maze = Arena.Graph()
maze.Build_Main()

cv2.imshow('top', grid.top)
cv2.waitKey(0)
cv2.destroyAllWindows()

key = {"TR" : 1, "SR" : 2, "CR" : 3, "TY" : 4, "SY" : 5, "CY" : 6}
pts = grid.Extract()
for i in range(len(pts)):
    maze.Assign(pts[i], i + 1)
    print(pts[i])

i = 0
tot = 100000
vis = 0
ok = False
link = True
add = True
done = False
(centre, direction, pos) = grid.Read_aruco()
start = pos
sim =  (-1, -1)
new = (-1, -1)

if pos == (5, 1):
    sim = (5, 3)
    new = (5, 4)
elif start == (5, 9):
    sim = (5, 7)
    new = (5, 6)
elif start == (1, 5):
    sim = (3, 5)
    new = (4, 5)
else:
    sim = (7, 5)
    new = (6, 5)

while i < tot:
    pts = grid.Extract()
    for i in range(len(pts)):
        maze.Assign(pts[i], i + 1)

    (centre, direction, pos) = grid.Read_aruco()
    print(centre, pos)

    val = grid.env.roll_dice()
    print("Target: ", val)
    val = key[val]

    ends = []
    for i in range(1, 10):
        for j in range(1, 10):
            if((i, j) != pos and (maze.matrix[i][j] == val)):
                  ends.append((i, j))
    
    # print(ends)
    opt = []
    l = 100
    for point in ends:
        path = maze.Bfs(pos, point)
        # print(path)
        if len(path) < l and path[-1][0] == point[0] and path[-1][1] == point[1]:
            l = len(path)
            opt = path
    
    print("Path: ", opt)
    print("\n")

    if l < 100: 
        last = centre
        for point in opt:
            vis += 1
            grid.Move(point)
            last = point

        if last == (5, 4):
            grid.Move((5, 5))
            ok = True
        elif last == (5, 6):
            grid.Move((5, 5))
            ok = True
        elif last == (6, 5):
            grid.Move((5, 5))
            ok = True
        elif last == (4, 5):
            grid.Move((4, 5))
            ok = True

    if ok:
        print('At the Centre')
        break 
        
    if link == True:
        maze.adj[start].pop(0)
        link = False

    if vis > 10 and done == False:
        maze.adj[sim].append(new)
        done = True
    
    if vis > 40 and add:
        add = False
        maze.adj[sim].pop(0)
        
    i += 1

