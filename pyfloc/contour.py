#!/usr/bin/env python

import time
import sys
import pickle
import functools
import numpy as np
import scipy as sp
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import splprep, splev
from sklearn import mixture
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from matplotlib.mlab import dist_point_to_segment
from scipy import misc 
from copy import deepcopy
import resource

resource.setrlimit(resource.RLIMIT_STACK, [resource.RLIM_INFINITY, resource.RLIM_INFINITY])
sys.setrecursionlimit(0x4000000)

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'indigo', 'silver', 'tomato', 'gold', 'springgreen', 'tan', 'cadetblue', 'aqua', 'khaki', 'indianred', 'brown', 'lime', 'ivory', 'lightsalmon', 'teal']
numerical_precision = 1e-10
np.set_printoptions(linewidth = np.inf)
print = functools.partial(print, flush=True)
def ccw(A,B,C):
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
def intersect(A,B,C,D):
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

class Polygon3D(object):
    """
    Polygons
    """
    def __init__(self, v, pdf = None):
        """
        Parameters
        ----------
        v : np.ndarray <number of vertexes> X 3
            An arbitrary number of vertexes can be used (>= 3)
        pdf : opened stream of a pdf file
        """
        if type(v) == list:
            self.v_ = np.array(v).astype(float)
        elif type(v) == np.ndarray:
            self.v_ = v.astype(float)
        self.n_v_ = np.shape(self.v_)[0] # number of vertexes
        self.c_ = self.center() # center
        self.i_, self.j_, self.n_ = self.tensors() # tensors on the plane of the surface (i_, j_) and normal to the surface
        self.d_ = np.dot(-self.n_,self.c_) # parameter d of the equation of the plane including the polygon  (ax+by+cz+d = 0)
        self.R_ = np.vstack((self.i_,self.j_,self.n_))
        self.v_ij_ = self.project_on_plane(self.v_)
        self.dv_ij = np.roll(self.v_ij_,1, axis = 0) - self.v_ij_ # array used by check_inside
        out_of_plane = np.linalg.norm(self.distance_from_plane_center(self.v_))
        if out_of_plane > numerical_precision:
            print('ERROR: the polygon element is not flat')
            print('\tout_of_plane = ',out_of_plane)
            print('\tself.v_ - self.c_ = ',self.v_ - self.c_)
            raise ValueError()
        if pdf != None:
            self.show(pdf)
    def center(self):
        return np.mean(self.v_, axis = 0)
    def tensors(self):
        tensor_x = self.v_[1,:] - self.v_[0,:]
        tensor_x /= np.linalg.norm(tensor_x)
        tensor_normal = np.cross(self.v_[1,:]-self.v_[0,:], self.v_[2,:]-self.v_[0,:])
        tensor_normal /= np.linalg.norm(tensor_normal)
        if (tensor_normal[2] < 0):
            tensor_normal = -tensor_normal
        elif (tensor_normal[2] == 0):
            if (tensor_normal[0] < 0):
                tensor_normal = -tensor_normal
            elif (tensor_normal[0] == 0):
                if (tensor_normal[1] < 0):
                    tensor_normal = -tensor_normal
        tensor_y = np.cross(tensor_normal, tensor_x)
        return tensor_x, tensor_y, tensor_normal
    def project_on_plane(self, points):
        if len(np.shape(points)) == 1:
            points = points.reshape(1,3)
        return np.dot(self.R_,(points-self.c_).transpose()).transpose()[:,:2]
    def distance_from_plane_center(self, points):
        if len(np.shape(points)) == 1:
            points = points.reshape(1,3)
        return np.dot(points-self.c_, self.n_)
    def check_inside(self, points):
        """
        Parameters
        ----------
        points : np.ndarray <Number of points> X 3

        Return
        ------
        np.ndarray <Number of points>
            True if the point projected on the plane of the polygon is inside the polygon
        """
        points_ij = self.project_on_plane(points)
        n_points = np.shape(points_ij)[0]
        flag_tmp1 = self.v_ij_[:,1].reshape((1,self.n_v_)) > points_ij[:,1].reshape((n_points,1))
        flag_tmp2 = np.roll(self.v_ij_[:,1],1).reshape((1,self.n_v_)) > points_ij[:,1].reshape((n_points,1))
        flag1 = flag_tmp1 != flag_tmp2
        flag_tmp1 = points_ij[:,1].reshape((n_points,1)) - self.v_ij_[:,1].reshape((1,self.n_v_))
        flag_tmp2 = self.v_ij_[:,0].reshape((1,self.n_v_)) + self.dv_ij[:,0].reshape((1,self.n_v_)) * flag_tmp1 / self.dv_ij[:,1].reshape((1,self.n_v_)) 
        flag2 = points_ij[:,0].reshape((n_points,1)) < flag_tmp2
        inside = np.mod(np.sum(np.logical_and(flag1,flag2), axis = 1), 2).astype(bool)
        return inside
    def test_check_inside(self, ax, n_points = 1000):
        points = np.vstack((np.random.uniform(np.min(self.v_[:,0]),np.max(self.v_[:,0]),n_points),np.random.uniform(np.min(self.v_[:,1]),np.max(self.v_[:,1]),n_points),np.random.uniform(np.min(self.v_[:,2]),np.max(self.v_[:,2]),n_points))).transpose()
        inside = self.check_inside(points)
        points_ij = self.project_on_plane(points)
        for i_point in range(n_points):
            if inside[i_point]:
                ax.plot(points_ij[i_point,0],points_ij[i_point,1],'.r')
            else:
                ax.plot(points_ij[i_point,0],points_ij[i_point,1],'.b')
    def calculate_plane_equation(self, points):
        return np.dot(self.n_.reshape(1,3),points.transpose())+self.d_
    def intersect_plane_line(self, points, line):
        """
        Parameters
        ----------
        points : np.ndarray
        line : np.array

        Return
        ------
        np.ndarray <Number of points> X 3
            3D coordinates of the point where the line passing through points intersect the plane of the polygon
        """
        den = np.dot(self.n_,line)
        if den == 0: # the line is orthogonal to the plane
            return np.nan*np.ones(np.shape(points))
        num = self.calculate_plane_equation(points)
        fract = np.dot(line.reshape(3,1),num).transpose() / den
        return points - fract
    def check_ray_crossing(self, points, ray):
        """
        The ray crosses the surface if:
            - The point lies below the plane of the surface
            - When the point is projected (in the direction of the ray) onto the plane of F, then it is inside F

        Parameters
        ----------
        points : np.ndarray
        ray : np.array

        Return
        ------
        np.ndarray <Number of points>
            True if the line starting from point in the direction ray intersect the plane of the polygon inside the polygon
        """
        if np.dot(self.n_,ray) == 0.0: # plane and ray are parallel --> no way of a crossing
            return np.zeros(np.shape(points)[0]).astype(bool)
        s_intersect = self.intersect_plane_line(points, ray)
        dist_from_plane = np.dot((points-s_intersect), ray)
        inside_surface = self.check_inside(s_intersect)
        #print 'DEBUG> s_intersect = ',s_intersect
        #print 'DEBUG> dist_from_plane = ',dist_from_plane
        #print 'DEBUG> inside_surface = ',inside_surface
        return np.logical_and(dist_from_plane < 0.0, inside_surface)
    def get_distance(self, points, cutoff = np.inf):
        """
        Parameters
        ----------
        points : np.ndarray
        cutoff : float

        Return
        ------
        float
            Distance with sign (above/below) normal
        """
        #print 'Calculating short-range for {0:d} points'.format(np.shape(points)[0])
        dist_sr = np.inf*np.ones(np.shape(points)[0])
        inds_points = np.arange(np.shape(points)[0]).astype(int)
        s_intersect = self.intersect_plane_line(points, self.n_)
        dist_from_plane = np.dot((points-s_intersect), self.n_)
        inds_close_to_plane = np.abs(dist_from_plane) < cutoff
        #print 'Points closer than cutoff from the plane: ',inds_close_to_plane
        inside_surface = self.check_inside(s_intersect[inds_close_to_plane,:])
        #print 'Points that project inside the polygon: ',inside_surface
        inds_sr = inds_points[inds_close_to_plane][inside_surface]
        dist_sr[inds_sr] = dist_from_plane[inds_sr]
        return dist_sr
    def plot3d(self, ax, color_lines = 'black'):
        """
        Parameters
        ----------
        ax : ax = fig.add_subplot(111, projection='3d')
        """
        for i_v in range(self.n_v_-1):
            ax.scatter(self.v_[i_v,0],self.v_[i_v,1],self.v_[i_v,2],'o',color = 'black')
            ax.scatter(self.v_[i_v+1,0],self.v_[i_v+1,1],self.v_[i_v+1,2],'o',color = 'black')
            ax.plot_wireframe([self.v_[i_v,0],self.v_[i_v+1,0]],[self.v_[i_v,1],self.v_[i_v+1,1]],[self.v_[i_v,2],self.v_[i_v+1,2]],color = color_lines)
        ax.plot_wireframe([self.v_[self.n_v_-1,0],self.v_[0,0]],[self.v_[self.n_v_-1,1],self.v_[0,1]],[self.v_[self.n_v_-1,2],self.v_[0,2]],color = color_lines)
    def plot(self, ax):
        for i_v in range(self.n_v_-1):
            ax.plot(self.v_ij_[i_v,0],self.v_ij_[i_v,1],'.k')
            plt.annotate(i_v, xy = self.v_ij_[i_v,:],xytext=(-10, 10),
                textcoords='offset points',
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
            ax.plot([self.v_ij_[i_v,0],self.v_ij_[i_v+1,0]],[self.v_ij_[i_v,1],self.v_ij_[i_v+1,1]],':k')
        ax.plot(self.v_ij_[i_v+1,0],self.v_ij_[i_v+1,1],'.k')
        plt.annotate(i_v+1, xy = self.v_ij_[i_v+1,:],xytext=(-10, 10),
            textcoords='offset points',
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
        ax.plot([self.v_ij_[self.n_v_-1,0],self.v_ij_[0,0]],[self.v_ij_[self.n_v_-1,1],self.v_ij_[0,1]],':k')
    def show(self, pdf):
        f = plt.figure()
        ax = f.add_subplot(111)
        self.plot(ax)
        pdf.savefig()
        plt.close()
    def write_vmd(self):
        output = ''
        for i_v in range(self.n_v_-1):
            output += 'draw line "{0:f}\t{1:f}\t{2:f}" "{3:f}\t{4:f}\t{5:f}"  style dashed\n'.format(self.v_[i_v,0],self.v_[i_v,1],self.v_[i_v,2],self.v_[i_v+1,0],self.v_[i_v+1,1],self.v_[i_v+1,2])
        output += 'draw line "{0:f}\t{1:f}\t{2:f}" "{3:f}\t{4:f}\t{5:f}"  style dashed\n'.format(self.v_[self.n_v_-1,0],self.v_[self.n_v_-1,1],self.v_[self.n_v_-1,2],self.v_[0,0],self.v_[0,1],self.v_[0,2])
        return output
    def parallel(self, other):
        return (1 - np.abs(np.dot(self.n_,other.n_))) < numerical_precision 
    def coplanar(self, other):
        if self.parallel(other):
            return np.abs(np.dot(self.c_ - other.c_, self.n_)) < numerical_precision 
        return False
    def contiguous(self, other):
        n_common = 0
        for i_v in range(self.n_v_):
            for j_v in range(other.n_v_):
                if np.linalg.norm(self.v_[i_v,:] - other.v_[j_v,:]) < numerical_precision:
                    n_common += 1
                    if n_common > 1:
                        return True
        return False
    def __eq__(self, other):
        v1 = [list(vertex) for vertex in list(self.v_)]
        v2 = [list(vertex) for vertex in list(other.v_)]
        v1.sort()
        v2.sort()
        if len(v1) != len(v2):
            return False
        v1 = np.array(v1)
        v2 = np.array(v2)
        return np.all(v1 == v2)
    def __ne__(self, other):
        return not self.__eq__(other)
    def __str__(self):
        output = 'Polygon element\n'
        for i_v in range(self.n_v_):
            output += '\tvertex[{0:d}] = {1:8.3f}\t{2:8.3f}\t{3:8.3f}\n'.format(i_v,self.v_[i_v,0],self.v_[i_v,1],self.v_[i_v,2])
        output += '\tTensors\n'
        output += '\t\ti = {0:8.3f}\t{1:8.3f}\t{2:8.3f}\n'.format(self.i_[0],self.i_[1],self.i_[2])
        output += '\t\tj = {0:8.3f}\t{1:8.3f}\t{2:8.3f}\n'.format(self.j_[0],self.j_[1],self.j_[2])
        output += '\t\tn = {0:8.3f}\t{1:8.3f}\t{2:8.3f}\n'.format(self.n_[0],self.n_[1],self.n_[2])
        output += '\tPlane equation: {0:+8.3f}*x  {1:+8.3f}*y {2:+8.3f}*z {3:+8.3f} = 0\n'.format(self.n_[0],self.n_[1],self.n_[2],self.d_)
        output += '\tcenter = {0:8.3f}\t{1:8.3f}\t{2:8.3f}\n'.format(self.c_[0],self.c_[1],self.c_[2])
        for i_v in range(self.n_v_):
            output += '\tvertex_projected[{0:d}] = {1:8.3f}\t{2:8.3f}\n'.format(i_v,self.v_ij_[i_v,0],self.v_ij_[i_v,1])
        return output[:-1]

class Contour(object):
    def __init__(self, data, n_bins = [100, 100], verbose = 0, outside_data = None):
        self.data = data
        self.outside_data = outside_data
        self.xc = None
        self.yc = None
        #--- Calculate histogram + Define grid
        min_x = np.min(self.data[:,0])
        max_x = np.max(self.data[:,0])
        dx = max_x - min_x
        min_y = np.min(self.data[:,1])
        max_y = np.max(self.data[:,1])
        dy = max_y - min_y
        self.H, self.xe, self.ye = np.histogram2d(self.data[:,0], self.data[:,1], bins = n_bins, range = [[min_x-0.1*dx, max_x+0.1*dx], [min_y-0.1*dy, max_y+0.1*dy]])
        self.H /= np.sum(self.H)
        self.xb = 0.5*(self.xe[:-1]+self.xe[1:])
        self.yb = 0.5*(self.ye[:-1]+self.ye[1:])
        X, Y = np.meshgrid(self.xb,self.yb)
        self.X = np.transpose(X)
        self.Y = np.transpose(Y)
        self.verbose = verbose
    def get_mask_inside_polygon(self):
        poly = self.get_polygon_refined()
        inside = poly.check_inside(points = np.vstack((self.X.flatten(), self.Y.flatten(), np.zeros(self.X.size))).transpose())
        return inside.reshape(self.X.shape)
    def get_index_inside_polygon(self):
        poly = self.get_polygon_refined()
        return poly.check_inside(points = np.vstack((self.data[:,0], self.data[:,1], np.zeros(self.data[:,0].size))).transpose())
    def intersect_polygon(self, x_new, y_new, ind, n_points, neighbour = True):
        """
        Check if by moving point ind to x_new, y_new we create a self intersecting polygon
        """
        ind_prev= (ind - 1) % (n_points - 1)
        ind_next= (ind + 1) % (n_points - 1)
        for i_now in range(n_points):
            i_now = i_now % (n_points - 1)
            if i_now not in [ind_prev, ind, ind_next]:
                i_next = (i_now + 1) % (n_points - 1)
                if i_next not in [ind_prev, ind, ind_next]:
                    if intersect((x_new, y_new), (self.xc[ind_prev], self.yc[ind_prev]), (self.xc[i_next], self.yc[i_next]), (self.xc[i_now], self.yc[i_now])):
                        return True
        if neighbour:
            ind_prev_prev = (ind - 2) % (n_points - 1)
            if intersect((x_new, y_new), (self.xc[ind], self.yc[ind]), (self.xc[ind_prev], self.yc[ind_prev]), (self.xc[ind_prev_prev], self.yc[ind_prev_prev])):
                return True
            ind_next_next = (ind + 2) % (n_points - 1)
            if intersect((x_new, y_new), (self.xc[ind], self.yc[ind]), (self.xc[ind_next], self.yc[ind_next]), (self.xc[ind_next_next], self.yc[ind_next_next])):
                return True
        return False
    def PolygonInteractor(self, stride = 0):
        if stride == 0:
            stride_inside = max(int(self.data.shape[0] / 20000),1)
            if self.outside_data is not None:
                stride_outside = max(int(self.outside_data.shape[0] / 20000),1)
            else:
                stride_outside = 1
        else:
            stride_inside = stride
            stride_outside = stride
        self.showverts = True
        self.epsilon = 5
        f = plt.figure()
        self.ax = f.add_subplot(111)
        from scipy.stats import binned_statistic_2d
        H, xe, ye, ix_iy = binned_statistic_2d(self.data[::stride_inside,0], self.data[::stride_inside,1], None, statistic = 'count', bins = 100, range = [[self.data[:,0].min(), self.data[:,0].max()],[self.data[:,1].min(), self.data[:,1].max()]], expand_binnumbers = True)
        ix_iy -= 1
        data_colors = H[ix_iy[0,:], ix_iy[1,:]]
        data_colors = np.log10(data_colors)
        if self.outside_data is not None:
            self.ax.plot(self.outside_data[::stride_outside,0], self.outside_data[::stride_outside,1], ',', color = 'dimgray')
        self.ax.scatter(self.data[::stride_inside,0], self.data[::stride_inside,1], marker = ',', s = 1.0, c = data_colors, cmap = 'inferno')
        #self.ax.pcolormesh(self.X, self.Y, self.H, cmap = plt.get_cmap('winter'))
        #self.ax.contour(self.X, self.Y, self.Z, cmap = plt.get_cmap('winter'))
        s = Polygon3D(np.vstack((self.xc, self.yc, np.zeros(self.xc.size))).transpose())
        inside = s.check_inside(points = np.vstack((self.X.flatten(), self.Y.flatten(), np.zeros(self.X.size))).transpose())
        inside = inside.reshape(self.X.shape)
        plt.title('Prob = {0:f}'.format(np.sum(self.H[inside])))
        self.poly = Polygon(np.column_stack([self.xc, self.yc]), animated = True, fill = False)
        if self.outside_data is not None:
            self.ax.set_xlim((min(np.min(self.outside_data[:,0]),np.min(self.data[:,0]), np.min(self.xe)), max(np.max(self.outside_data[:,0]),np.max(self.data[:,0]),np.max(self.xe))))
            self.ax.set_ylim((min(np.min(self.outside_data[:,1]),np.min(self.data[:,1]), np.min(self.ye)), max(np.max(self.outside_data[:,1]),np.max(self.data[:,1]),np.max(self.ye))))
        else:
            self.ax.set_xlim((min(np.min(self.data[:,0]), np.min(self.xe)), max(np.max(self.data[:,0]),np.max(self.xe))))
            self.ax.set_ylim((min(np.min(self.data[:,1]), np.min(self.ye)), max(np.max(self.data[:,1]),np.max(self.ye))))
        self.ax.add_patch(self.poly)
        canvas = self.poly.figure.canvas
        x, y = zip(*self.poly.xy)
        self.line = Line2D(x, y,
                           marker='o', markerfacecolor='r',
                           animated=True)
        self.ax.add_line(self.line)
        self.cid = self.poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert
        canvas.mpl_connect('draw_event', self.draw_callback)
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('key_press_event', self.key_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        self.canvas = canvas
    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        # do not need to blit here, this will fire before the screen is updated
    def poly_changed(self, poly):
        'this method is called whenever the polygon object is called'
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state
        self.xc, self.yc = zip(*self.poly.xy)
        s = Polygon3D(np.vstack((self.xc, self.yc, np.zeros(self.xc.size))).transpose())
        inside = s.check_inside(points = np.vstack((self.X.flatten(), self.Y.flatten(), np.zeros(self.X.size))).transpose())
        inside = inside.reshape(self.X.shape)
        plt.title('Prob = {0:f}'.format(np.sum(self.H[inside])))
    def get_ind_under_point(self, event):
        'get the index of the vertex under point if within epsilon tolerance'
        # display coords
        xy = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.hypot(xt - event.x, yt - event.y)
        indseq, = np.nonzero(d == d.min())
        ind = indseq[0]
        if d[ind] >= self.epsilon:
            ind = None
        return ind
    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        if not self.showverts:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)
    def button_release_callback(self, event):
        'whenever a mouse button is released'
        if not self.showverts:
            return
        if event.button != 1:
            return
        self._ind = None
    def key_press_callback(self, event):
        'whenever a key is pressed'
        if not event.inaxes:
            return
        if event.key == 't':
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            if not self.showverts:
                self._ind = None
        elif event.key == 'd':
            ind = self.get_ind_under_point(event)
            if ind is not None:
                self.poly.xy = np.delete(self.poly.xy,
                                         ind, axis=0)
                self.line.set_data(zip(*self.poly.xy))
        elif event.key == 'i':
            xys = self.poly.get_transform().transform(self.poly.xy)
            p = event.x, event.y  # display coords
            for i in range(len(xys) - 1):
                s0 = xys[i]
                s1 = xys[i + 1]
                d = dist_point_to_segment(p, s0, s1)
                if d <= self.epsilon:
                    self.poly.xy = np.insert(
                        self.poly.xy, i+1,
                        [event.xdata, event.ydata],
                        axis=0)
                    self.line.set_data(zip(*self.poly.xy))
                    break
        if self.line.stale:
            self.canvas.draw_idle()
        self.f.canvas.draw()
        self.f.canvas.flush_events()
    def motion_notify_callback(self, event):
        'on mouse movement'
        if not self.showverts:
            return
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata
        self.poly.xy[self._ind] = x, y
        if self._ind == 0:
            self.poly.xy[-1] = x, y
        elif self._ind == len(self.poly.xy) - 1:
            self.poly.xy[0] = x, y
        self.line.set_data(zip(*self.poly.xy))
        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)
    def get_polygon_refined(self):
        self.xc, self.yc = zip(*self.poly.xy)
        return Polygon3D(np.vstack((self.xc, self.yc, np.zeros(len(self.xc)))).transpose())

class Apple(Contour):
    """
    Attributes
    ----------
    data:   np.ndarray
        Samples
    H:  np.ndarray
        Probability histogram
    Z:  np.ndarray
        Fitting of the probability histrogram with gaussian functions
    T:  np.ndarray
        T = (Z - (value of the contour line of Z that encapsulate the desired probability of H))**2
        This is the function that is minimized when searching the contour
    G:  list of np.ndarray
        Gradient of T
    X:  np.ndarray
        Mesh values along x
    Y:  np.ndarray
        Mesh values along y
    xb: np.ndarray
        Grid bin centers along x
    yb: np.ndarray
        Grid bin centers along y
    """
    def __init__(self, data, n_bins = [100, 100], n_gaussians = 1, prob_target = 0.1):
        """
        Parameters
        ----------
        data:   np.ndarray
            shape: <n_samples> x 2
        n_gaussians: int
            Number of gaussians used to fit the distribution of samples
        prob_target:  float
            The gate will include the region with this probability
        n_bins: list
            The number of bins along x and y
        """
        #--- Fit distributions with multiple gaussians
        super(Apple,self).__init__(data, n_bins)
        clf = mixture.GaussianMixture(n_components = n_gaussians, covariance_type = 'full')
        clf.fit(self.data)
        XY = np.array([self.X.ravel(), self.Y.ravel()]).T
        self.Z = clf.score_samples(XY)
        self.Z = self.Z.reshape(self.X.shape)
        self.Z -= np.min(self.Z) 
        self.Z /= np.sum(self.Z)
        #--- Required points
        n_peaks = 1
        while True:
            inds = np.argsort(self.H.flatten())[-n_peaks:]
            prob = np.sum(self.H.flatten()[inds])
            if prob > 0.5*prob_target:
                break
            n_peaks += 1
            if n_peaks >= np.prod(n_bins):
                raise ValueError('ERROR')
        #i_x_peaks, i_y_peaks = np.unravel_index(np.argsort(self.H.flatten())[-n_peaks:], self.Z.shape)
        i_x_peaks, i_y_peaks = np.unravel_index(inds, self.Z.shape)
        self.x_peaks = self.xb[i_x_peaks]
        self.y_peaks = self.yb[i_y_peaks]
        #--- Plots
        f = plt.figure()
        ax1 = f.add_subplot(221)
        ax1.plot(self.data[:,0], self.data[:,1], ',')
        ax1.pcolormesh(self.X, self.Y, self.H, cmap = plt.get_cmap('hot'))
        plt.title('Probability')
        ax2 = f.add_subplot(222)
        ax2.pcolormesh(self.X, self.Y, self.Z, cmap = plt.get_cmap('hot'))
        cax = ax2.contour(self.X, self.Y, self.Z, levels = np.linspace(np.min(self.Z), np.max(self.Z),100), cmap = plt.get_cmap('winter'))
        #--- Grid bin size
        min_x = np.min(self.data[:,0])
        max_x = np.max(self.data[:,0])
        dx = max_x - min_x
        min_y = np.min(self.data[:,1])
        max_y = np.max(self.data[:,1])
        dy = max_y - min_y
        dxb = self.xb[1] - self.xb[0]
        dyb = self.yb[1] - self.yb[0]
        n_points = 40
        outer_x = np.empty(n_points) 
        outer_y = np.empty(n_points) 
        outer_x[:int(0.25*n_points)] = np.linspace(min_x+1*dxb,max_x-1*dxb,int(0.25*n_points))
        outer_y[:int(0.25*n_points)] = min_y
        outer_x[int(0.25*n_points):int(0.5*n_points)] = max_x
        outer_y[int(0.25*n_points):int(0.5*n_points)] = np.linspace(min_y+1*dyb,max_y-1*dyb,int(0.5*n_points)-int(0.25*n_points))
        outer_x[int(0.5*n_points):int(0.75*n_points)] = np.linspace(max_x-1*dxb,min_x+1*dxb,int(0.75*n_points)-int(0.5*n_points))
        outer_y[int(0.5*n_points):int(0.75*n_points)] = max_y
        outer_x[int(0.75*n_points):] = min_x
        outer_y[int(0.75*n_points):] = np.linspace(max_y-1*dyb,min_y+1*dyb,n_points-int(0.75*n_points))
        tck, u = splprep(np.vstack((outer_x,outer_y)), u = None, s = 0.0, per = 1)
        u_new = np.linspace(u.min(), u.max(), n_points)
        outer_x, outer_y = splev(u_new, tck, der=0)
        border = Polygon3D(np.vstack((outer_x, outer_y, np.zeros(outer_x.size))).transpose())
        dist_target = np.inf
        for i_level, segs in enumerate(cax.allsegs):
            prob_inside = 0.0
            for i_seg, seg in enumerate(segs):
                if len(seg) > 2:
                    inside = border.check_inside(np.vstack((seg[:,0], seg[:,1], np.zeros(seg[:,0].size))).transpose())
                    if np.sum(inside) == len(inside):
                        #ax2.plot(seg[:,0],seg[:,1],':k')
                        s = Polygon3D(np.vstack((seg[:,0], seg[:,1], np.zeros(seg[:,0].size))).transpose())
                        inside = s.check_inside(points = np.vstack((self.X.flatten(), self.Y.flatten(), np.zeros(self.X.size))).transpose())
                        inside = inside.reshape(self.X.shape)
                        prob_inside += np.sum(self.H[inside])
            if np.abs(prob_inside-prob_target) < dist_target:
                dist_target = np.abs(prob_inside-prob_target)
                i_target = i_level
        print('Best contour {0:d} at {1:f}'.format(i_target,cax.levels[i_target]))
        for i_seg, seg in enumerate(cax.allsegs[i_target]):
            ax2.plot(seg[:,0],seg[:,1],':r')
        ax2.plot(outer_x,outer_y,'o--r')
        if len(cax.allsegs[i_target]) == 1:
            print('Setting contour at {0:f}'.format(cax.levels[i_target]))
            self.xc = cax.allsegs[i_target][0][:,0]
            self.yc = cax.allsegs[i_target][0][:,1]
        else:
            from scipy.spatial import ConvexHull
            points = np.empty((0,2))
            for seg in cax.allsegs[i_target]:
                points = np.vstack((points,seg))
            hull = ConvexHull(points)
            self.xc = points[hull.vertices,0]
            self.yc = points[hull.vertices,1]
        ax2.plot(self.x_peaks, self.y_peaks,'.k')
        plt.title('Gaussian fit')
        #--- Define normalized target function to minimize
        self.T = np.power(self.Z - cax.levels[i_target],2.0)
        self.T /= np.max(self.T)
        #--- Define normalized gradien of target function
        self.G = np.gradient(self.T)
        max_G = max(np.max(np.abs(self.G[0])),np.max(np.abs(self.G[1])))
        self.G[0] = self.G[0] / max_G
        self.G[1] = self.G[1] / max_G
        ax3 = f.add_subplot(234)
        cax = ax3.pcolormesh(self.X, self.Y, np.log10(self.T), cmap = plt.get_cmap('hot'))
        plt.title('Cost function')
        f.colorbar(cax)
        ax4 = f.add_subplot(235)
        ax4.pcolormesh(self.X, self.Y, self.G[0], cmap = plt.get_cmap('bwr'))
        plt.title('Gradient y')
        ax5 = f.add_subplot(236)
        ax5.pcolormesh(self.X, self.Y, self.G[1], cmap = plt.get_cmap('bwr'))
        plt.title('Gradient x')
        plt.show()
    def run(self, n_points = 10, max_iter = 0, stride_show = np.inf, tol = 1e-3):
        """
        Parameters
        ----------
        n_points:   int
            Number of points in the contour
        max_iter:   int
            Maximum number of iterations
        stride_show:    int
            Plot contour every stride_show iterations
        tol:    float
            Stop when the sum of the target function along the contour is below tol
        """
        #--- Grid bin size
        min_x = np.min(self.data[:,0])
        max_x = np.max(self.data[:,0])
        dx = max_x - min_x
        min_y = np.min(self.data[:,1])
        max_y = np.max(self.data[:,1])
        dy = max_y - min_y
        dxb = self.xb[1] - self.xb[0]
        dyb = self.yb[1] - self.yb[0]
        delta_move = 1.0*np.sqrt(dxb**2.0 + dyb**2.0)
        perc_forced_inside = 0.9
        n_intersect = 0
        n_superimpose = 0
        n_delete = 0
        n_reject = 0
        n_uphill = 0
        n_missing = 0
        #--- Define initial contour = external boundary
        if self.xc is None:
            self.xc = np.empty(n_points) 
            self.yc = np.empty(n_points) 
            self.xc[:int(0.25*n_points)] = np.linspace(min_x+1*dxb,max_x-1*dxb,int(0.25*n_points))
            self.yc[:int(0.25*n_points)] = min_y
            self.xc[int(0.25*n_points):int(0.5*n_points)] = max_x
            self.yc[int(0.25*n_points):int(0.5*n_points)] = np.linspace(min_y+1*dyb,max_y-1*dyb,int(0.5*n_points)-int(0.25*n_points))
            self.xc[int(0.5*n_points):int(0.75*n_points)] = np.linspace(max_x-1*dxb,min_x+1*dxb,int(0.75*n_points)-int(0.5*n_points))
            self.yc[int(0.5*n_points):int(0.75*n_points)] = max_y
            self.xc[int(0.75*n_points):] = min_x
            self.yc[int(0.75*n_points):] = np.linspace(max_y-1*dyb,min_y+1*dyb,n_points-int(0.75*n_points))
        #--- Calculate the spline curve fitting the contour
        tck, u = splprep(np.vstack((self.xc,self.yc)), u = None, s = 0.0, per = 1)
        u_new = np.linspace(u.min(), u.max(), n_points)
        self.xc, self.yc = splev(u_new, tck, der=0)
        #--- Optimize contour
        i_iter = 0
        total_T = np.inf
        best_T = np.inf
        xc_best = self.xc
        yc_best = self.yc
        print('Gate optimization')
        while (i_iter < max_iter) and (total_T > tol):
            xc_old = np.copy(self.xc)
            yc_old = np.copy(self.yc)
            #--- Log show
            if (i_iter > 0) and (i_iter % stride_show) == 0:
                print('\ti_iter {0:d} total_T {1:f} best_T {2:f}'.format(i_iter, total_T, best_T))
                print('\tn_intersect = ',n_intersect,' n_superimpose = ',n_superimpose,' n_delete = ',n_delete,' n_reject = ',n_reject,' n_uphill = ',n_uphill,' n_missing = ',n_missing)
                f = plt.figure()
                ax1 = f.add_subplot(231)
                ax1.pcolormesh(self.X, self.Y, self.H, cmap = plt.get_cmap('hot'))
                ax1.contour(self.X, self.Y, self.Z, cmap = plt.get_cmap('winter'))
                ax1.plot(self.data[:,0], self.data[:,1], ',')
                ax1.plot(self.xc,self.yc,'o--r')
                ax1.plot(xc_best,yc_best,'o--g')
                plt.title('Histogram')
                ax2 = f.add_subplot(232)
                s = Polygon3D(np.vstack((self.xc, self.yc, np.zeros(self.xc.size))).transpose())
                inside = s.check_inside(points = np.vstack((self.X.flatten(), self.Y.flatten(), np.zeros(self.X.size))).transpose())
                inside = inside.reshape(self.X.shape)
                ax2.pcolormesh(self.X, self.Y, inside, cmap = plt.get_cmap('cool'))
                ax2.plot(self.x_peaks, self.y_peaks,'.k')
                plt.title('Probability = {0:f}'.format(np.sum(self.H[inside])))
                ax3 = f.add_subplot(233)
                cax = ax3.pcolormesh(self.X, self.Y, np.log10(self.T), cmap = plt.get_cmap('hot'))
                ax3.plot(self.xc,self.yc,'o--b')
                plt.title('Cost function')
                f.colorbar(cax)
                ax3 = f.add_subplot(223)
                ax3.pcolormesh(self.X, self.Y, self.G[0], cmap = plt.get_cmap('bwr'))
                plt.title('Gradient y')
                ax4 = f.add_subplot(224)
                ax4.pcolormesh(self.X, self.Y, self.G[1], cmap = plt.get_cmap('bwr'))
                plt.title('Gradient x')
                plt.show()
            #--- Collect statistics along the contour
            i_bins = []
            total_T = 0
            for i in range(n_points):
                i_xc = np.argmin(np.abs(self.xb-self.xc[i]))
                i_yc = np.argmin(np.abs(self.yb-self.yc[i]))
                i_bins.append((i_xc, i_yc))
                total_T += self.T[i_xc, i_yc]
            if total_T < best_T:
                best_T = total_T
                xc_best = np.copy(self.xc)
                yc_best = np.copy(self.yc)
            #print('i_iter {0:d} total_T {1:f} best_T {2:f}'.format(i_iter, total_T, best_T))
            #--- Select a point
            ind = i_iter % n_points
            xs = self.xc[ind]
            ys = self.yc[ind]
            i_xs, i_ys = i_bins[ind]
            #--- Move the point
            grd_x = self.G[1][i_xs,i_ys]
            grd_y = self.G[0][i_xs,i_ys]
            move_x = delta_move * (-0.0*grd_x + 1.0*np.random.randn())
            move_y = delta_move * (-0.0*grd_y + 1.0*np.random.randn())
            xs_new = min(max_x,max(min_x, xs+move_x))
            ys_new = min(max_y,max(min_y, ys+move_y))
            i_xs_new = np.argmin(np.abs(self.xb-(xs_new)))
            i_ys_new = np.argmin(np.abs(self.yb-(ys_new)))
            i_xs_new = min(self.X.shape[0]-1,max(0, i_xs_new))
            i_ys_new = min(self.X.shape[1]-1,max(0, i_ys_new))
            #--- If it didn't move, delete the point (The rational is that it was in a low-gradient region)
            if (i_xs_new == i_xs) and (i_ys_new == i_ys):
                self.xc = []
                self.yc = []
                for i in range(n_points):
                    if i != ind:
                        self.xc.append(xc_old[i])
                        self.yc.append(yc_old[i])
                #--- Check if deleting the point we excluded the required peaks
                tck, u = splprep(np.vstack((self.xc,self.yc)), u = None, s = 0.0, per = 1)
                u_new = np.linspace(u.min(), u.max(), n_points)
                self.xc, self.yc = splev(u_new, tck, der=0)
                s = Polygon3D(np.vstack((self.xc, self.yc, np.zeros(self.xc.size))).transpose())
                inside = s.check_inside(points = np.vstack((self.x_peaks, self.y_peaks, np.zeros(self.x_peaks.size))).transpose())
                if (np.sum(inside) < perc_forced_inside*len(inside)): # in case go back to previous contour
                    self.xc = xc_old
                    self.yc = yc_old
                i_iter += 1
                n_delete += 1
                continue
            #--- If it superimposed with other points, reject the movement
            if (i_xs_new, i_ys_new) in i_bins:
                i_iter += 1
                n_superimpose += 1
                continue
            #--- If it caused polygon intersections, reject the movement
            if self.intersect_polygon(self.xb[i_xs_new], self.yb[i_ys_new], ind, n_points):
                n_intersect += 1
                i_iter += 1
                continue
            #--- Move the point
            self.xc[ind] = self.xb[i_xs_new]
            self.yc[ind] = self.yb[i_ys_new]
            #--- Sanity check: remove intersections
            xc_no_intersection = []
            yc_no_intersection = []
            for ind in range(n_points-1):
                if not self.intersect_polygon(self.xc[ind], self.yc[ind], ind, n_points, False):
                    xc_no_intersection.append(self.xc[ind])
                    yc_no_intersection.append(self.yc[ind])
            self.xc = xc_no_intersection
            self.yc = yc_no_intersection
            #--- Calculate the spline curve fitting the contour
            tck, u = splprep(np.vstack((self.xc,self.yc)), u = None, s = 0.0, per = 1)
            u_new = np.linspace(u.min(), u.max(), n_points)
            self.xc, self.yc = splev(u_new, tck, der=0)
            s = Polygon3D(np.vstack((self.xc, self.yc, np.zeros(self.xc.size))).transpose())
            inside = s.check_inside(points = np.vstack((self.x_peaks, self.y_peaks, np.zeros(self.x_peaks.size))).transpose())
            #--- Check if the movement excluded the point of maximum density, if so go back
            if (np.sum(inside) < perc_forced_inside*len(inside)): # in case go back to previous contour
                self.xc = xc_old
                self.yc = yc_old
                n_missing += 1
                i_iter += 1
                continue
            #--- Calculate how the movement changed the target function
            delta = (self.T[i_xs_new,i_ys_new] - self.T[i_xs,i_ys])
            accept = False
            if delta <= 0.0: # if the movement decreases it, accept the new point
                accept = True
            elif np.random.rand() > np.exp(-delta): # othersize accept with probability p
                accept = True
                n_uphill += 1
            else:
                n_reject += 1
            #--- If not accepted, go back to previous contoir
            if not accept:
                self.xc = xc_old
                self.yc = yc_old
            i_iter += 1
        print('Final target function value {0:f}'.format(best_T))
        self.xc = xc_best
        self.yc = yc_best
        self.PolygonInteractor()
        plt.show()

class Cherry(Contour):
    def __init__(self, data = None, outside_data = None, n_bins = [100, 100], verbose = 0):
        """
        Parameters
        ----------
        H: numeric matrix. Could be an image or an histogram2d  
        data: [data_x,data_y] list of numpy arrays you want to be in x,y axes when calculating numpy histogram
        bins: [bin_x,bin_y] list to specify bins for numpy histogram
	nbins: int, number of bins
        starting_point: [x,y] list of the coordinates of the starting point or the center as numpy.array([x,y])
        eclude_borders: useful for cluster analysis, because after normalization sometimes data tend to clump in the borders
        xc, yc: coordinates of the analogic bins of the contour

        Notes
        -----
        Either H or data,bins,starting_point have not to be None 

        The Cherry matrix will be built:
        0: points above threshold
        1: points below threshold
        2: points inside the external contour
        3: points outside the external contour
        4: -----------for further improvements-----------------
        5: points belonging to the contour set
        6: points belonging to the external contour
        """
        super(Cherry,self).__init__(data, n_bins, verbose, outside_data = outside_data)
    def analToDigit(self, anal_point):
        xbins = self.bins[0]
        ybins = self.bins[1]
        #find the bin in which the event belongs to
        x=(anal_point[0]-xbins)<0
        if(x[0]==True): #mi sa che non c'e' necessita' di questo if, il primo elemento non potra' mai essere True
            x=0
        else:
            #-1 because the correct position is the last False
            x=np.where(x)[0][0]-1
        y=(anal_point[1]-ybins)<0
        if(y[0]==True):
            y=0
        else:
            y=np.where(y)[0][0]-1
        return np.array([x,y])
    def digitToAnal(self, digi_point):
        #transform the digital center into analogic, and the new coordinates
        # are those of the bin center the point belongs to
    
                            #################
                            #       #       #
                            #   O   #       #
                            #       #       #
                            #################
        xbins = self.bins[0]
        ybins = self.bins[1]
        x=(xbins[digi_point[0]]+xbins[digi_point[0]+1])/2
        y=(ybins[digi_point[1]]+ybins[digi_point[1]+1])/2
        return np.array([x,y])
    def cherry(self, x, y, diagonals = False):
        """
        Description
        -----------
        Performs the analysis of the Valid zone, finding the contours of the analyzed image

        Parameters
        ----------
        x: x coordinate of the analyzed point
        y: y coordinate of the analyzed point
        diagonals: whether to consider the diagonal points as neighbors or not
        """
        if self.M[x][y]==0:
            self.M[x][y]=2
            if x!=0:
                self.cherry(x-1, y, diagonals)
            if y!=0:
                self.cherry(x, y-1, diagonals)
            if x!=len(self.M)-1:
                self.cherry(x+1, y, diagonals)
            if y!=len(self.M[0])-1:
                self.cherry(x, y+1, diagonals)
            if diagonals == True:
                if x!=0 and y!=0:
                    self.cherry(x-1, y-1, diagonals) 
                if x!=0 and y!=len(self.M[0])-1:
                    self.cherry(x-1, y+1, diagonals)
                if x!=len(self.M)-1 and y!=len(self.M[0])-1:
                    self.cherry(x+1, y+1, diagonals)
                if x!=len(self.M)-1 and y!=0:
                    self.cherry(x+1, y-1, diagonals)   
        elif self.M[x][y]==1: 
            self.M[x][y]=5 
    def fill_gaps(self, x, y, diagonals = False):
        """
        Description
        -----------
        Performs the analysis of the Invalid zone, finding the extern contour of the analyzed image

        Parameters
        ----------
        x: x coordinate of the analyzed point
        y: y coordinate of the analyzed point
        diagonals: whether to consider the diagonal points as neighbors or not
        """
        if self.M[x][y]<2:
            self.M[x][y]=3
            if x!=0:
                self.fill_gaps(x-1, y, diagonals)
            if y!=0:
                self.fill_gaps(x, y-1, diagonals)
            if x!=len(self.M)-1:
                self.fill_gaps(x+1, y, diagonals)
            if y!=len(self.M[0])-1:
                self.fill_gaps(x, y+1, diagonals)
            if diagonals == True:
                if x!=0 and y!=0:
                    self.fill_gaps(x-1, y-1, diagonals) 
                if x!=0 and y!=len(self.M[0])-1:
                    self.fill_gaps(x-1, y+1, diagonals)
                if x!=len(self.M)-1 and y!=len(self.M[0])-1:
                    self.fill_gaps(x+1, y+1, diagonals)
                if x!=len(self.M)-1 and y!=0:
                    self.fill_gaps(x+1, y-1, diagonals)   
        if self.M[x][y]==5:
            self.M[x][y]=6 

            if diagonals == True:
                if x!=0:
                    self.fill_gaps(x-1, y, diagonals)
                if y!=0:
                    self.fill_gaps(x, y-1, diagonals)
                if x!=len(self.M)-1:
                    self.fill_gaps(x+1, y, diagonals)
                if y!=len(self.M[0])-1:
                    self.fill_gaps(x, y+1, diagonals)
    def define_starting_point(self, M_value = 0, y_simple=False, random = False):
        """
        Description
        -----------
        Defines the starting point to start analysis in the left border of the image 

        Parameters
        ----------
        M_value: the zone which allows to calculate the starting point
        y_simple: whether the starting point is referring to the simple contour definition or to the zone analysis
        random: whether to randomly choose the starting point (inside a certain zone) or to use the first valid starting point

        Notes
        -----
        If there are more than a picture in the image, setting random = False, the function will take the coordinates of the first
        valid pixel of the first image it finds
        """
        if random == False:
            x = np.where(self.M==M_value)[0][0]
            y = np.where(self.M==M_value)[1][0]
        else:  
            index = np.random.randint(0,len(np.where(self.M==0)[0]))
            x = np.where(self.M==M_value)[0][index]
            y = np.where(self.M==M_value)[1][index]
        if y_simple == True:
            y=0
        return [x,y]
    def move_to_contour(self, x, y):
        """
        Description
        -----------
        From the starting point it moves until it reaches the contour, redefining the starting point 

        Parameters
        ----------
        M: Cherry matrix
        x: x coordinate of the analyzed point
        y: y coordinate of the analyzed point
        """
        while (self.M[x][y+1]==1):
            y=y+1
        return x,y
    def simple_contour_definition(self, x, y, dist = 0):
        """
        Description
        -----------
        Defines the external contour of the analyzed image 

        Parameters
        ----------
        M: Cherry matrix
        x: x coordinate of the analyzed point
        y: y coordinate of the analyzed point
        dist: distance from the previous pixel
        """
        if self.M[x][y]==1 and dist <=1:
            dist = dist +1
            if x!=0 and self.M[x-1][y]==0:
                self.M[x][y]=6
                dist=0
            if y!=0 and self.M[x][y-1]==0:
                self.M[x][y]=6
                dist=0
            if x!=len(self.M)-1 and self.M[x+1][y]==0:
                self.M[x][y]=6
                dist=0
            if y!=len(self.M[0])-1 and self.M[x][y+1]==0:
                self.M[x][y]=6
                dist=0
            if x!=0:
                self.simple_contour_definition(x-1, y, dist)
            if y!=0:
                self.simple_contour_definition(x, y-1, dist)
            if x!=len(self.M)-1:
                self.simple_contour_definition(x+1, y, dist)
            if y!=len(self.M[0])-1:
                self.simple_contour_definition(x, y+1, dist)
    def take_inside(self):
        """
        Description
        -----------
        Take inside the set of inner points some set of points that are inside the external contour:
        0: valid elements inside a contour of invalid elements
        1: invalid elements
        5: inner contour
        
        Parameters
        ----------
        M: Cherry matrix
        """
        self.M[self.M==0]=2 
        self.M[self.M==1]=2 
        self.M[self.M==5]=2 
    def adjust_borders(self, contour_valid = True):
        """
        Description
        -----------
        Decides whether the external contour belongs to the set of valid or invalid elements
        
        Parameters
        ----------
        contour_valid: if set to true, the extern contour is considered as valid
        """
        if(contour_valid): 
            self.M[self.M==6]=2
        else:
            self.M[self.M==6]=3 
    def delete_outsiders(self):
        """
        Description
        -----------
        Deletes the points outside the external contour 
        """
        self.M[self.M==3]=0
    def adjust_contour_borders(self):
        """
        Description
        -----------
        Adjust the external contour to include valid points in the border of the image
        """
        self.M[:,-1][np.where(self.M[:,-1]==2)[0]]=6 
        self.M[:,0][np.where(self.M[:,0]==2)[0]]=6
        self.M[0,:][np.where(self.M[0,:]==2)[0]]=6
        self.M[-1,:][np.where(self.M[-1,:]==2)[0]]=6 
    def check_count(self, threshold, count_threshold=0.3, mode='above'):
        """
        Description
        -----------
        Checks if enough elements are inside the contour. If not, it modifies the matrix H to
        consider as valids the neighbors of the already valid elements. In this case, the
        analysis should be repeated (for cluster analysis)
        
        Parameters
        ----------
        count_threshold: lower bound threshold which determines how many points are enough 
        """
        check = 0
        taken = 1.0*len(np.where(self.M==2)[0]) + 1.0*len(np.where(self.M==6)[0])
        total = 1.0*len(np.where(self.H_filled >= threshold)[0])
        if taken < count_threshold*total:
            self.M = np.zeros(self.H_filled.shape)
            if mode=='above':
                self.M[self.H_filled < threshold] = 1
            elif mode=='below':
                self.M[self.H_filled > threshold] = 1
            rows = np.where(self.M==0)[0]
            columns = np.where(self.M==0)[1]
            for i in range(len(rows)):
                self.explode_H(rows[i], columns[i], threshold)
            check = 1
        return check
    def explode_H(self, x, y, threshold):
        """
        Description
        -----------
        Modifies the analyzed point to consider valid also its neighbors (for cluster analysis)
        
        Parameters
        ----------
        x: x coordinate of the analyzed point
        y: y coordinate of the analyzed point
        """
        if x!=0 and self.H_filled[x-1][y]<threshold:
            self.H_filled[x-1][y] = threshold 
            self.M[x-1][y]=0
        if x!=0 and y!=0 and self.H_filled[x-1][y-1]<threshold:
            self.H_filled[x-1][y-1]=threshold 
            self.M[x-1][y-1]=0
        if y!=0 and self.H_filled[x][y-1]<threshold:
            self.H_filled[x][y-1]=threshold
            self.M[x][y-1]=0
        if x!=0 and y!=len(self.M[0])-1 and self.H_filled[x-1][y+1]<threshold:
            self.H_filled[x-1][y+1]=threshold
            self.M[x-1][y+1]=0
        if x!=len(self.M)-1 and self.H_filled[x+1][y]<threshold:
            self.H_filled[x+1][y]=threshold
            self.M[x+1][y]=0
        if x!=len(self.M)-1 and y!=len(self.M[0])-1 and self.H_filled[x+1][y+1]<threshold:
            self.H_filled[x+1][y+1]=threshold
            self.M[x+1][y+1]=0
        if y!=len(self.M[0])-1 and self.H_filled[x][y+1]<threshold:
            self.H_filled[x][y+1]=threshold
            self.M[x][y+1]=0
        if x!=len(self.M)-1 and y!=0 and self.H_filled[x+1][y-1]<threshold:
            self.H_filled[x+1][y-1]=threshold
            self.M[x+1][y-1]=0
    def get_contour(self, mode = 'external'):
        """
        Description
        -----------
        Returns a matrix with the contour elements
            
        Parameters
        ----------
        mode:
        'external': take the external contour
        'internal': take the internal contours
        'all': take all contours
        """     
        M=deepcopy(self.M)
        if mode == 'external':
            M[M<6] = 0
        elif mode == 'internal':
            M[M!=5] = 0
        elif mode == 'all':
            M[M<5] = 0
        return M
    def get_contour_ordered(self):
        """
        Description
        -----------
        find the coordinates xc, yc of the bins of the contour (ordered)
        """
        contour = self.get_contour()
        john_rando = np.random.randint(0,len(np.where(contour==6)[0]))
        x = np.where(contour==6)[0][john_rando]
        y = np.where(contour==6)[1][john_rando]
        self.xc = []
        self.yc = []
        self.order_contour(contour, x, y)
        self.xc = np.array(self.xb[self.xc])
        self.yc = np.array(self.yb[self.yc])
    def order_contour(self, contour, x, y):  
        if x!=0 and contour[x-1][y]==6:
            contour[x-1][y] = 4 #checked
            self.xc.append(x-1)
            self.yc.append(y)
            self.order_contour(contour, x-1, y)
        if x!=0 and y!=0 and contour[x-1][y-1]==6:
            contour[x-1][y-1] = 4 
            self.xc.append(x-1)
            self.yc.append(y-1)
            self.order_contour(contour, x-1, y-1)
        if y!=0 and contour[x][y-1]==6:
            contour[x][y-1] = 4
            self.xc.append(x)
            self.yc.append(y-1)
            self.order_contour(contour, x, y-1)
        if x!=len(contour)-1 and y!=0 and contour[x+1][y-1]==6:
            contour[x+1][y-1] = 4 
            self.xc.append(x+1)
            self.yc.append(y-1)
            self.order_contour(contour, x+1, y-1)
        if x!=len(contour)-1 and contour[x+1][y]==6:
            contour[x+1][y] = 4
            self.xc.append(x+1)
            self.yc.append(y)
            self.order_contour(contour, x+1, y)
        if x!=len(contour)-1 and y!=len(contour)-1 and contour[x+1][y+1]==6:
            contour[x+1][y+1] = 4 
            self.xc.append(x+1)
            self.yc.append(y+1)
            self.order_contour(contour, x+1, y+1)
        if y!=len(contour)-1 and contour[x][y+1]==6:
            contour[x][y+1] = 4
            self.xc.append(x)
            self.yc.append(y+1)
            self.order_contour(contour, x, y+1)
        if x!=0 and y!=len(contour)-1 and contour[x-1][y+1]==6:
            contour[x-1][y+1] = 4 
            self.xc.append(x-1)
            self.yc.append(y+1)
            self.order_contour(contour, x-1, y+1)
    def get_contour_np(self, mode = 'external'):
        """
        Description
        -----------
        Returns a boolean array that defines which data elements belong to the contour
        [T, F, F, ...]
            
        Parameters
        ----------
        mode:
        'external': take the external contour
        'internal': take the internal contours
        'all': take all contours
        'internal_point': take the internal points of the external contour
        """
        M=deepcopy(self.M)
        if mode == 'external':
            M[M<6] = 0
        elif mode == 'internal':
            M[M!=5] = 0
        elif mode == 'all':
            M[M<5] = 0
        elif mode == 'internal_points':
            M[M!=2] = 0
        xbins = self.bins[0]
        ybins = self.bins[1]
        contour = np.zeros(np.shape(self.data)[1])
        M = np.where(M)
        for i in range(len(M[0])):
            x=M[0][i]
            y=M[1][i]
            anal = np.array(self.digitToAnal([x,y], self.bins, self.nbins))
            #the transformed point is located in the center of the bin,
            #but the original point could have been located among the entire bin
            deviationx = (xbins[x+1]-xbins[x])/2
            deviationy = (ybins[y+1]-ybins[y])/2
            deviation = np.array([deviationx,deviationy])
            #look in the dataframe where the found point (index) is,
            #creating a bool vector
            cond = abs(self.data - anal[:, np.newaxis]) <= deviation[:, np.newaxis]
            cond = (cond[0] * cond[1]).astype(bool)
            #If contour[i] is not 0, the element in that position will be chosen
            contour=contour+cond
        return contour.astype(bool)
    def run(self, prob_target = 0.9, starting_point = None, exclude_borders = False, mode = 'above', n_points = 20, take_inside = True, diagonals = False, min_threshold = 0.1):
        """
        Description
        -----------
        Performs the complete analysis, defining all the zones of the image
        
        Parameters
        ----------
        take_inside:
            bool
            decides if all the elements inside the external contour will be taken 
        diagonals:
            bool
            whether to consider the diagonal points as neighbors or not
        min_threshold: 
            float in [0,1] 
            to be considered as valid contour, it has to respect the threshold of the taken valid points over the total valid points
                        
        Notes
        -----
        starting point: for cluster analysis, I suggest to use the coordinates of the bin with the highest density
        of the cluster you want to analyze
        """

        for threshold in np.sort(self.H.flatten())[-1::-10]:
            self.M = np.zeros(np.shape(self.H), dtype=int)
            self.H_filled = np.copy(self.H)
            if mode=='above':
                self.M[self.H_filled < threshold] = 1
            if mode=='below':
                self.M[self.H_filled > threshold] = 1
            n_bins_selected = np.sum(self.M == 0)
            if n_bins_selected == 0:
                continue
            if starting_point is None:
                self.starting_point = self.define_starting_point(random=True)
            else: 
                if type(starting_point) is np.ndarray: 
                    self.starting_point = self.analToDigit(starting_point)
                else:    
                    self.starting_point = starting_point
            check = 1
            while (check):
                x = self.starting_point[0]
                y = self.starting_point[1]
                if self.verbose > 2:
                    print('\nOriginal histogram:\n', self.H)
                    print('\nCorresponding Cherry matrix M:\n', self.M)
                    print('\nStarting point: ', [x,y])
                self.cherry(x,y, diagonals)
                if self.verbose > 2:
                    print('\nM after filling:\n', self.M)
                self.fill_gaps(0,0, diagonals)
                self.fill_gaps(0,len(self.M[0])-1, diagonals)
                self.fill_gaps(len(self.M)-1,0, diagonals)
                self.fill_gaps(len(self.M)-1,len(self.M[0])-1, diagonals)
                if self.verbose > 2:
                    print('\nM after the external contour is found:\n', self.M)
                if take_inside:
                    self.take_inside()
                    if self.verbose > 2:
                        print ('\nM after all the elements inside the external contour are taken:\n', self.M)
                check = self.check_count(threshold, count_threshold = min_threshold, mode = 'above' )
            self.adjust_contour_borders()
            if self.verbose > 2:
                print('\nM after the valid points in the border of the image are taken as part of the contour:\n', self.M)
            prob_inside = np.sum(self.H[self.M == 2])
            if self.verbose > 2:
                print('Probability: {0:f} / {1:f}'.format(prob_inside,np.sum(self.H)))
            if prob_inside >= prob_target:
                break
        self.get_contour_ordered()
        # Interpolate the numerical contour over n_points
        tck, u = splprep(np.vstack((self.xc,self.yc)), u = None, s = 0.0, per = 1)
        u_new = np.linspace(u.min(), u.max(), n_points)
        self.xc, self.yc = splev(u_new, tck, der=0)
        self.PolygonInteractor()
        plt.show()
    def run_just_contour(self):
        """
        Description
        -----------
        Finds the external contour of an image
             
        """
        if self.verbose > 1:
            print('Original histogram:\n', self.H)
            print('\nCorresponding Cherry matrix M:\n', self.M)
        x = self.starting_point[0]
        y = 0
        x,y = self.move_to_contour(x,y)
        self.simple_contour_definition(x,y)
        if self.verbose > 2:
            print('\nM after the external contour is found:\n', self.M)
    def get_polygon_refined(self):
        self.xc, self.yc = zip(*self.poly.xy)
        poly = Polygon3D(np.vstack((self.xc, self.yc, np.zeros(len(self.xc)))).transpose())
        points = np.vstack((self.X.flatten(), self.Y.flatten(), np.zeros(len(self.X.flatten())))).transpose()
        inds = poly.check_inside(points = points)
        inds = inds.reshape(self.X.shape)
        self.M = np.zeros(self.M.shape)
        self.M[inds] = 2
    def get_index_inside_polygon(self, data = None):
        if data is None:
            data = self.data
        M = deepcopy(self.M)
        M[M!=2] = 0
        from scipy.stats import binned_statistic_2d
        dummy_H, dummy_xe, dummy_ye, ix_iy = binned_statistic_2d(data[:,0], data[:,1], None, statistic = 'count', bins = [self.xe, self.ye], expand_binnumbers = True)
        ix_iy -= 1
        # everything outside the M grid is reported to the first bin (assuming that it's False)
        ix_iy[0,ix_iy[0,:] < 0] = 0
        ix_iy[0,ix_iy[0,:] >= len(self.xe)-1] = 0
        ix_iy[1,ix_iy[1,:] < 0] = 0
        ix_iy[1,ix_iy[1,:] >= len(self.ye)-1] = 0
        M = M.astype(bool)
        M[0,0] = False
        return M[ix_iy[0,:],ix_iy[1,:]]
    def get_mask_inside_polygon(self):
        M = deepcopy(self.M)
        M[M!=2] = 0
        return M.astype(bool)

def make_data(kind, n_samples = 1000, n_samples_rare = 10):
    """
    Generate toy data sets for testing the algorithm

    Parameters
    ----------
    n_samples:  int
        Number of samples per class
    n_samples_rare:  int
        Number of samples per rare classes
    """
    from sklearn import datasets
    if kind == 'circles':
        X,y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
    elif kind == 'moons':
        X,y = datasets.make_moons(n_samples=n_samples, noise=.05)
    elif kind == 'blobs':
        X,y = datasets.make_blobs(n_samples = n_samples, centers = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], cluster_std = [0.25, 0.25, 0.25])
    elif kind == 'gates':
        X = np.array([0.25,0.25]) + 0.05*np.random.randn(n_samples,2)
        x = np.array([0.25,0.75]) + 0.05*np.random.randn(n_samples,2)
        X = np.vstack((X,x))
        x = np.hstack((np.random.uniform(low = 0.0, high = 1.0, size = (n_samples_rare,1)),np.random.uniform(low = 0.0, high = 1.0, size = (n_samples_rare,1))))
        X = np.vstack((X,x))
    else:
        raise ValueError('ERROR: {0:s} kind does not exist'.format(kind))
    return X

if __name__ == '__main__':
    print('------------------')
    print('Testing contour.py')
    print('------------------')

    #pdf = PdfPages('./test.pdf')

    X = make_data('blobs', 10000, 100)

    #C = Apple(X, n_gaussians = 10, prob_target = 0.9, n_bins = [100,100])
    #C.run(n_points = 20, max_iter = 10000, stride_show = 5000, tol = 1e-1)
    #p = C.get_polygon_refined()

    C = Cherry(data = X, n_bins = [100, 100], verbose = 2)
    C.run(prob_target = 0.6, exclude_borders = False, mode='above')
    C.get_polygon_refined()

    #pdf.close()
