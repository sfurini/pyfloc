#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt

def is_red(x):
    return np.all(x == np.array([1,0,0,1]))

class PointInteractor(object):
    """
    Select/deselect points on a scatter plot
    """
    def __init__(self, x, y, inds = []):
        """
        Parameters
        ----------
        x: np.ndarray
        y: np.ndarray
        inds: list
            Indexes of samples selected before Interactor
            These are plotted in red, the other ones in black
        """
        if len(x) != len(y):
            return ValueError('ERROR: wrong dimensions for x and y')
        colors = ['black']*len(x)
        self.inds = np.zeros(len(x))
        for i in inds:
            colors[i] = 'red'
            self.inds[i] = 1
        self.f = plt.figure()
        ax = self.f.add_subplot(111)
        plt.title('Click for selecting points')
        self.points = ax.scatter(x, y, color =  colors, picker = 1)
    def on_pick(self, event):
        if self.inds[event.ind[0]] == 0:
            # black --> red
            self.points._facecolors[event.ind,:] = (1, 0, 0, 1)
            self.points._edgecolors[event.ind,:] = (1, 0, 0, 1)
            self.inds[event.ind[0]] = 1
            #print('switching ',event.ind[0],' to red')
        else:
            # red --> black
            self.points._facecolors[event.ind,:] = (0, 0, 0, 1)
            self.points._edgecolors[event.ind,:] = (0, 0, 0, 1)
            self.inds[event.ind[0]] = 0
            #print('switching ',event.ind[0],' to black')
        self.f.canvas.draw()
        self.f.canvas.flush_events()
    def key_press_callback(self, event):
        if not event.inaxes:
            return
        if event.key == 'x':
            plt.xscale('linear')
        elif event.key == 'X':
            plt.xscale('log')
        elif event.key == 'y':
            plt.yscale('linear')
        elif event.key == 'Y':
            plt.yscale('log')
        self.f.canvas.draw()
        self.f.canvas.flush_events()
    def run(self):
        self.f.canvas.mpl_connect('pick_event', self.on_pick)
        self.f.canvas.mpl_connect('key_press_event', self.key_press_callback)
        plt.show()
        self.f.canvas.flush_events()
    def get_actives(self):
        return np.where(self.inds)[0]

class AxesScaleInteractor(object):
    """
    Change lin/log scale on x-y axes

    For figures with subplots, it only changes the scale of the last subplot made
    """
    def __init__(self, figure):
        self.f = figure
        self.f.canvas.mpl_connect('key_press_event', self.key_press_callback)
        #self.f.canvas.mpl_connect('axes_enter_event', self.enter_axes)
        plt.show()
    #def enter_axes(self, event):
    #    print('enter_axes', event.inaxes)
    #    event.inaxes.patch.set_facecolor('yellow')
    #    self.f.canvas.draw()
    def key_press_callback(self, event):
        if not event.inaxes:
            return
        self.f.sca(event.inaxes)
        if event.key == 'x':
            plt.xscale('linear')
        elif event.key == 'X':
            plt.xscale('log')
        elif event.key == 'y':
            plt.yscale('linear')
        elif event.key == 'Y':
            plt.yscale('log')
        self.f.canvas.draw()

class PolygonInteractor(object):
    """
    A polygon editor

    Key-bindings
      't' toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them
      'd' delete the vertex under point
      'i' insert a vertex at point.  You must be within epsilon of the
          line connecting two existing vertices
    """
    showverts = True
    epsilon = 5  # max pixel distance to count as a vertex hit
    def __init__(self, ax, poly):
        if poly.figure is None:
            raise RuntimeError('You must first add the polygon to a figure '
                               'or canvas before defining the interactor')
        self.ax = ax
        canvas = poly.figure.canvas
        self.poly = poly
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

if __name__ == '__main__':
    print('-------------------')
    print('Testing graphics.py')
    print('-------------------')

    x = [1,10,100,1000]
    y = [1,10,100,1000]
    x = [1,2,3,4]
    y = [1,2,3,4]

    i = [1]
    pi = PointColorInteractor(x,y,i)
    pi.run()
    print(pi.get_actives())

