import numpy as np
from scipy.integrate import solve_ivp
import pygame_render
import pygame

def flatten(t):
    return [item for sublist in t for item in sublist]


class BalanceRobot():
    def __init__(self, dt):
        self.p = {'r': 0.05,
                  'l': 0.15,
                  'm1': 0.5,
                  'm2': 0.1,
                  'g': 9.81,
                  'I1': 0.006,
                  'I2': 0.002}

        self.canvas_w = 1
        self.canvas_h = 1

        self.step_time = dt
        self.y0 = [0, 0.001, 0, 0]
        self.render = pygame_render.Render(caption='Balance robot')
        self.w, self.h = self.render.get_resolution()

        w_px = int(0.08*self.w*self.canvas_w)
        l_px = int(2*self.p['l']*self.h*self.canvas_h)
        self.r_px = int(self.p['r'] * self.h * self.canvas_h)
        b_h = 800
        b_w = 800

        surf = pygame.Surface((b_w, b_h), pygame.SRCALPHA)
        points = [((b_w-w_px)//2, b_w//2), ((b_w-w_px)//2, (b_h-l_px)//2),
                  ((b_w+w_px)//2, (b_h-l_px)//2), ((b_w+w_px)//2, b_w//2)]
        _ = pygame.draw.polygon(surface=surf, color=(200, 200, 200, 150), points=points)
        self.body = pygame_render.RenderObject(surf=surf, x=self.w//2, y=self.h-self.r_px)

        self.render.add_object(key='body', layer='1', obj=self.body)

        surf2 = pygame.Surface((b_w, b_h), pygame.SRCALPHA)
        _ = pygame.draw.circle(surface=surf2, color=(255, 150, 150, 150), center=(b_w//2, b_h//2),
                               radius=int(self.p['r']*self.h))
        _ = pygame.draw.line(surface=surf2, color=(0, 0, 0, 150), start_pos=(b_w//2-int(self.p['r']*self.h), b_h//2),
                             end_pos=(b_w // 2 + int(self.p['r']*self.h), b_h // 2), width=3)

        self.wheel = pygame_render.RenderObject(surf=surf2, x=self.w//2, y=self.h-self.r_px)
        self.render.add_object(key='wheel', layer='2', obj=self.wheel)


    def step(self, torque):
        disturbance = np.random.normal(0, 0.0)
        dx = self.balance_sim(0, self.y0, torque)

        sol = solve_ivp(fun=self.balance_sim, t_span=[0, self.step_time], y0=self.y0, method='RK45',
                        args=[torque],
                        dense_output=False, t_eval=[self.step_time])


        self.y0 = flatten(sol.y)

        # for rendering only
        self.body.set_angle(self.y0[1])
        x = -int(self.y0[3] * self.p['r'] * self.w)
        self.body.set_pos(x=self.w//2 + x, y=self.h-self.r_px)
        self.wheel.set_angle(self.y0[3])
        self.wheel.set_pos(x=self.w//2 + x, y=self.h-self.r_px)
        self.render.update_window(fps=int(1 / self.step_time))
        return dx[0], dx[2]

    def get_state(self):
        pos = -self.p['r'] * self.y0[3] - self.p['l'] * np.sin(self.y0[1])
        vel = -self.p['r'] * self.y0[2] - self.p['l'] * np.cos(self.y0[1]) * self.y0[0]
        states = {'da1': self.y0[0],
                  'a1': self.y0[1],
                  'da2': self.y0[2],
                  'a2': self.y0[3],
                  'vel': vel,
                  'pos': pos}
        return states

    def balance_sim(self, t, X, M):
        # X = [do1, o1, do2, o2]
        do1 = X[0]
        o1 = X[1]
        do2 = X[2]
        o2 = X[3]
        r = self.p['r']
        l = self.p['l']
        m1 = self.p['m1']
        m2 = self.p['m2']
        g = self.p['g']
        I1 = self.p['I1']
        I2 = self.p['I2']

        ddo2 = (-M - (m1*r*l*np.cos(o1) * (M + l*m1*g*np.sin(o1))/(m1*(l**2)+I1)) + m1*r*l*np.sin(o1)*(do1**2)) / \
               (m1*r**2 - ((r*l*m1*np.cos(o1))**2)/(m1*l**2 + I1) + m2*r**2 + I2)

        ddo1 = (M - m1*r*ddo2*l*np.cos(o1) + m1*g*l*np.sin(o1))/(m1*l**2 + I1)
        dX = [ddo1, do1, ddo2, do2]
        return dX


