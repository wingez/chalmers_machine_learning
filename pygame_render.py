import pygame
import numpy as np

class Render(object):
    def __init__(self, caption, resolution=(1000, 1000)):
        self.window = pygame.display.set_mode(resolution)
        pygame.display.set_caption(caption)
        self.w = resolution[0]
        self.h = resolution[1]
        self.render_layers = dict()
        self.clock = pygame.time.Clock()

    def add_object(self, key, layer, obj):
        # check if layer exists in dict, layer and key should be strings.
        if layer not in self.render_layers:
            self.render_layers[layer] = dict()
        self.render_layers[layer][key] = obj

    def remove_object(self, key):
        for i, (layer, value) in enumerate(self.render_layers.items()):
            if key in value:
                self.render_layers[layer].pop(key, None)
                return

    def check_key_exists(self, key, layer=None):
        if layer is None:
            for i, (layer, value) in enumerate(self.render_layers):
                if key in value:
                    return True
        else:
            for v in self.render_layers[layer]:
                if key in v:
                    return True
        return False

    def get_layer(self, layer):
        if layer not in self.render_layers:
            return dict()
        return self.render_layers[layer]

    def get_object(self, key):
        for i, (layer, value) in enumerate(self.render_layers.items()):
            if key in value:
                return self.render_layers[layer][key], layer
        return None, None

    def update_object(self, obj, key, layer=None):
        if layer is None:
            for i, (layer, value) in enumerate(self.render_layers.items()):
                if key in value:
                    self.render_layers[layer][key] = obj
        self.render_layers[layer][key] = obj

    def update_window(self, fps=None):
        self.window.fill((100, 100, 100))
        for i, (layer, value) in enumerate(self.render_layers.items()):
            for j, (key, obj) in enumerate(value.items()):
                self.window.blit(obj.get_rot_surface(), obj.get_pos())
        pygame.display.update()
        if fps is not None:
            self.clock.tick(fps)

    def get_resolution(self):
        return self.w, self.h

    def get_main_window(self):
        return self.window


class RenderObject(object):
    def __init__(self, surf, x, y):
        '''
        :param surf:
        :param x: Horizontal (top left (0,0))
        :param y: Vertical
        '''
        self.surf = surf
        self.rot_surf = surf
        self.x = x
        self.y = y
        self.pos = self.surf.get_rect(center=(x, y))
        self.angle = 0

    def set_angle(self, angle):
        self.angle = 180*angle/np.pi

    def get_rot_surface(self):
        self.rot_surf = pygame.transform.rotate(self.surf, self.angle)
        return self.rot_surf

    def get_surface(self):
        return self.surf

    def update_surface(self, surface):
        self.surf = surface

    def get_pos(self):
        return self.rot_surf.get_rect(center=(self.x, self.y))

    def get_pos_xy(self):
        return self.x, self.y

    def set_pos(self, x, y):
        self.x = x
        self.y = y
