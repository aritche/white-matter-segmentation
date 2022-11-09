from visdom import Visdom
import numpy as np

# Class from https://github.com/noagarcia/visdom-tutorial
class VisdomLinePlotter(object):
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    
    def plot(self, y_var_name, split_name, title_name, x, y, x_var_name='Epochs'):
        if y_var_name not in self.plots:
            options = {
                'legend': [split_name],
                'title': title_name,
                'xlabel': x_var_name,
                'ylabel': y_var_name
            }
            self.plots[y_var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=options)
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[y_var_name], name=split_name, update='append')
