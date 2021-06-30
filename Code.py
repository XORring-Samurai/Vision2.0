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
        # p.stepSimulation()

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



    def Distance (self, A, B):
        """
        Method to find Manhattan Distance b/w 2 points.
        Arguments:
            2 tuples b/w which, we need to find the distance.
        Return Value:
            The Distance b/w the given points.
        """
        return abs(A[0] - B[0]) + abs(A[1] - B[1])


    
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
        return angle

    
    def Rotate ():
        pass
    
    def Move ():
        pass


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
                # red squares
                epsilon_s = 0.17 * cv2.arcLength(contour, True)
                approx_s = cv2.approxPolyDP(contour, epsilon_s, True)
                if len(approx_s) == 4:
                    cv2.drawContours(blank_red, [approx_s], -1, 255, -1)

                    M = cv2.moments(approx_s)
                    if M['m00'] != 0:
                        cx = M['m10'] / M['m00']
                        cy = M['m01'] / M['m00']
                        cx = int(cx)
                        cy = int(cy)
                        rs.append([cx, cy])
                
                # red triangles
                epsilon_t = 0.12 * cv2.arcLength(contour, True)
                approx_t = cv2.approxPolyDP(contour, epsilon_t, True)
                if len(approx_t) == 3:
                    cv2.drawContours(blank_red, [approx_t], -1, 150, -1)

                    M = cv2.moments(approx_t)
                    if M['m00'] != 0:
                        cx = M['m10'] / M['m00']
                        cy = M['m01'] / M['m00']
                        cx = int(cx)
                        cy = int(cy)
                        rt.append([cx, cy])
                
                # red circles
                epsilon_c = 0.01 * cv2.arcLength(contour, True)
                approx_c = cv2.approxPolyDP(contour, epsilon_c, True)
                if len(approx_c) > 4:
                    cv2.drawContours(blank_red, [approx_c], -1, 200, -1)

                    M = cv2.moments(approx_c)
                    if M['m00'] != 0:
                        cx = M['m10'] / M['m00']
                        cy = M['m01'] / M['m00']
                        cx = int(cx)
                        cy = int(cy)
                        rc.append([cx, cy])

        for contour in contours_yellow:
            ar = cv2.contourArea(contour)
            if ar > 100: 
                # yellow squares
                epsilon_s = 0.17 * cv2.arcLength(contour, True)
                approx_s = cv2.approxPolyDP(contour, epsilon_s, True)
                if len(approx_s) == 4:
                    cv2.drawContours(blank_yellow, [approx_s], -1, 255, -1)

                    M = cv2.moments(approx_s)
                    if M['m00'] != 0:
                        cx = M['m10'] / M['m00']
                        cy = M['m01'] / M['m00']
                        cx = int(cx)
                        cy = int(cy)
                        ys.append([cx, cy])
                
                # yellow triangles
                epsilon_t = 0.12 * cv2.arcLength(contour, True)
                approx_t = cv2.approxPolyDP(contour, epsilon_t, True)
                if len(approx_t) == 3:
                    cv2.drawContours(blank_yellow, [approx_t], -1, 150, -1)

                    M = cv2.moments(approx_s)
                    if M['m00'] != 0:
                        cx = M['m10'] / M['m00']
                        cy = M['m01'] / M['m00']
                        cx = int(cx)
                        cy = int(cy)
                        yt.append([cx, cy])

                # yellow circles
                epsilon_c = 0.01 * cv2.arcLength(contour, True)
                approx_c = cv2.approxPolyDP(contour, epsilon_c, True)
                if len(approx_c) > 4:
                    cv2.drawContours(blank_yellow, [approx_c], -1, 200, -1)

                    M = cv2.moments(approx_c)
                    if M['m00'] != 0:
                        cx = M['m10'] / M['m00']
                        cy = M['m01'] / M['m00']
                        cx = int(cx)
                        cy = int(cy)
                        yc.append([cx, cy])

        rs = Transform(rs)
        rt = Transform(rt)
        rc = Transform(rc)
        ys = Transform(ys)
        yt = Transform(yt)
        yc = Transform(yc)

        return (rs, rt, rc, ys, yt, yc)
        

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

            return path
            
grid = Arena()
bot_pos = grid.Read_aruco()
dist = grid.Distance((1, 1), (3, 3))
pts = grid.Extract()
cv2.imshow('grid', grid.top)
cv2.waitKey(0)
cv2.destroyAllWindows()

maze = Arena.Graph()
for i in range(len(pts)):
    maze.Assign(pts[i], i + 1)
maze.Build_Main()
print(maze.matrix[1: , 1:])
for i in range(1, 10):
    for j in range(1, 10):
        print(i, j, "=", maze.adj[(i, j)])

path = maze.Bfs((5, 1), (5, 1))
print(path)