import numpy as np

from scipy.stats import skewnorm, norm
import pygame_render
import pygame


class Environment(object):
    def __init__(self, map_type, fps=5, resolution=(1000, 1000)):
        self.map_type = map_type
        self.fps = fps
        self.position = np.zeros(2)
        self.renderer = pygame_render.Render(caption='Mining planet 42', resolution=resolution)
        pygame.font.init()
        init_renderer_layers(self.renderer)
        load_backdrop(map_type, self.renderer)
        self.grid_size, self.tile_res_w, self.tile_res_h = setup_renderer(self.renderer, self.map_type)
        map_ = self.create_map()
        self.actor = Actor(map_, np.zeros(2, dtype=int))
        pos = self.actor.position
        tile_res = self.actor.map.tile_res
        tx = int(tile_res[0])
        ty = int(tile_res[1])
        surf = pygame.Surface((int(1.5*tx), int(1.5*ty)), pygame.SRCALPHA)

        self.img_robot = {'N': pygame.transform.scale(pygame.image.load("imgs/robot_n.png").convert_alpha(),
                                                      (int(1.5*tx), int(1.5*ty))),
                          'S': pygame.transform.scale(pygame.image.load("imgs/robot_s.png").convert_alpha(),
                                                      (int(1.5*tx), int(1.5*ty))),
                          'E': pygame.transform.scale(pygame.image.load("imgs/robot_e.png").convert_alpha(),
                                                      (int(1.5*tx), int(1.5*ty))),
                          'W': pygame.transform.scale(pygame.image.load("imgs/robot_w.png").convert_alpha(),
                                                      (int(1.5*tx), int(1.5*ty)))}
        surf.blit(self.img_robot["N"], (0,0))
        obj = pygame_render.RenderObject(surf=surf, x=0, y=0)
        self.renderer.add_object(key='robot', layer='robot', obj=obj)
        self.properties = None
        self.ground_truth = None
        self.sensor_properties = ['ground_density', 'moist', 'reflectivity', 'silicon_rate', 'oxygen_rate', 'iron_rate',
                                  'aluminium_rate', 'magnesium_rate', 'undetectable']
        self.accuracy_tree = 0
        self.accuracy_knn = 0
        self.img_deposit = [pygame.transform.scale(pygame.image.load("imgs/hotspot.png").convert_alpha(), (tx, ty)),
                            pygame.transform.scale(pygame.image.load("imgs/hotspot1.png").convert_alpha(), (tx, ty))]
        self.k = 0



        if map_type == 2:
            self.plt_acc = PLT_ACC(self.renderer, self.tile_res_w, self.tile_res_h)

    def create_map(self):
        map_ = Map(map_size=np.array([self.grid_size, self.grid_size]),
                   tile_res=np.array([self.tile_res_w, self.tile_res_h]),
                   non_valid_transitions=init_non_valid_transitions(self.map_type),
                   init_minerals=initialize_minerals(self.map_type, self.grid_size),
                   altitude_map=np.zeros((self.grid_size, self.grid_size)),
                   temp_map=np.zeros((self.grid_size, self.grid_size)))
        return map_

    def step(self, action):
        self.properties, self.ground_truth = self.actor.step(action, True)

    def get_actor(self):
        return self.actor

    def render(self):
        self.k += 1
        update_render_objects(self.renderer, self.actor, self.img_deposit, self.img_robot, self.k % 2)
        self.renderer.update_window(self.fps)

    def get_sensor_readings(self):
        return self.properties

    def get_sensor_properties(self):
        return self.sensor_properties

    def get_ground_truth(self):
        return self.ground_truth

    def exit(self):
        pygame.quit()

    def get_action_space(self):
        return self.actor.action_space

    def update_accuracy(self, acc_tree, acc_knn):
        self.accuracy_tree = acc_tree
        self.accuracy_knn = acc_knn

def init_renderer_layers(render):
    layers = ['backdrop', 'grid', 'temp_map', 'mineral_deposit', 'robot']
    for key in layers:
        render.render_layers[key] = dict()

def update_render_objects(render, actor, img_deposit, robot_img, k_):
    ## update temp map
    # delete previous surface
    render.remove_object("temp_map")
    # add updated surface
    temp_obj = create_temp_obj(render, actor)
    render.add_object(obj=temp_obj, key='temp_map', layer='temp_map')

    # update minerals
    # get layer
    render_minerals = render.get_layer(layer='mineral_deposit')
    minerals = actor.map.minerals

    # remove if it has been removed
    key_pop_list = []
    for i, (k, v) in enumerate(render_minerals.items()):
        if k not in minerals:
            key_pop_list.append(k)

    for k in key_pop_list:
        render.remove_object(k)

    tile_res = actor.map.tile_res
    tx = int(tile_res[0])
    ty = int(tile_res[1])
    surf = pygame.Surface((tx, ty), pygame.SRCALPHA)

    surf.blit(img_deposit[k_], (0, 0))
    # add if not already in layer
    for i, (k, v) in enumerate(minerals.items()):
        pos = str_to_tuple(k)
        obj = pygame_render.RenderObject(surf=surf, x=int(2+tile_res[0]*(pos[0]+0.5)), y=int(2+tile_res[1]*(pos[1]+0.5)))
        render.add_object(obj=obj, key=k, layer='mineral_deposit')

    # update pose of robot
    robot, _ = render.get_object(key='robot')
    pos = actor.position
    robot.set_pos(x=int(2+tile_res[0]*(pos[0]+0.5)), y=int(2+tile_res[1]*(pos[1]+0.5)))
    surf = pygame.Surface((int(1.5 * tx), int(1.5 * ty)), pygame.SRCALPHA)
    surf.blit(robot_img[actor.direction], (0, 0))
    robot.update_surface(surf)

def create_temp_obj(render, actor):
    ambient_temp = actor.map.ambient_temp
    max_temp = 1000
    temp_map = actor.map.temperature_map
    tile_res = actor.map.tile_res
    w, h = render.get_resolution()
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    temp_map_shape = temp_map.shape
    for x in range(temp_map_shape[1]):
        for y in range(temp_map_shape[1]):
            alpha = int(min([(256*(temp_map[y, x] - ambient_temp)/max_temp), 256]))
            color = (255, 0, 0, alpha)
            rect = pygame.Rect(3+int(tile_res[0]*x), 3+int(tile_res[1]*y), int(tile_res[0]), int(tile_res[1]))
            pygame.draw.rect(surface=surf, color=color, rect=rect)

    obj = pygame_render.RenderObject(surf=surf, x=w//2, y=h//2)
    return obj

def init_non_valid_transitions(map_type):
    non_valid = set()
    if map_type == 1:
        pass

    elif map_type == 2:
        pass
    return non_valid


def initialize_minerals(map_type, map_size):
    minerals = dict()
    return minerals


def setup_renderer(render, map_type):
    w, h = render.get_resolution()

    if map_type == 1:
        grid_size = 6
    else:
        grid_size = 25

    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    color = (255, 255, 255, 80)

    horizontal_lines, tile_res_w = grid_spacer(side_margin_px=2, map_size=grid_size, map_res=w)
    for i in range(len(horizontal_lines)):
        start_pos = (0, horizontal_lines[i])
        end_pos = (w, horizontal_lines[i])
        pygame.draw.line(surface=surf, color=color, start_pos=start_pos, end_pos=end_pos, width=2)

    vertical_lines, tile_res_h = grid_spacer(side_margin_px=2, map_size=grid_size, map_res=h)
    for i in range(len(vertical_lines)):
        start_pos = (vertical_lines[i], 0)
        end_pos = (vertical_lines[i], h)
        pygame.draw.line(surface=surf, color=color, start_pos=start_pos, end_pos=end_pos, width=2)

    obj = pygame_render.RenderObject(surf=surf, x=w//2, y=h//2)
    render.add_object(key='grid', layer='grid', obj=obj)
    return grid_size, tile_res_w, tile_res_h


def grid_spacer(side_margin_px, map_size, map_res):
    grid_res = map_res-3*side_margin_px
    tile_res = grid_res/map_size
    res = []
    for i in range(map_size+1):
        res.append(side_margin_px + int(np.round(i*tile_res)))
    return res, tile_res


class Actor(object):
    def __init__(self, map_, position):
        self.map = map_
        self.position = position
        self.action_space = ['N', 'S', 'W', 'E']
        self.n_actions = len(self.action_space)
        self.direction = 'N'

    def step(self, action, new_points=False):
        pygame.event.get()
        new_pos, score = self.map.transition(pos=self.position, action=action)
        self.direction = action
        self.map.update(new_points=new_points)

        self.position = new_pos
        if new_points:
            parameter, mineral = self.map.sample(self.position)
            return parameter, mineral
        else:
            return score



class Map(object):
    def __init__(self, map_size, tile_res, non_valid_transitions, init_minerals, altitude_map, temp_map):
        self.map_size = map_size
        self.tile_res = tile_res
        self.altitude_map = altitude_map
        self.non_valid_transitions = non_valid_transitions
        self.minerals = init_minerals
        self.temperature_map = temp_map
        self.hot_spots = dict()
        self.temperature_limit = 150
        self.p_new_hot_spot = 0.01*map_size[0]*map_size[1]
        self.hot_spot_emerge_time = 4
        self.temp_decay_factor = 0.8
        self.minerals_decay_time = 15
        self.ambient_temp = 40
        self.mineral_spawn_temp = 500

    def get_score(self, pos):
        score = 0
        # Is there a mineral?
        key = str(tuple(pos))
        if key in self.minerals:
            score += 25

        # Is there too much heat?
        index = tuple(pos)
        temp = self.temperature_map[index]
        if temp > self.temperature_limit:
            score += -2 - (temp - self.temperature_limit)/10

        closest_mineral = np.inf

        for key in self.minerals:
            pos_mineral = str_to_tuple(key)
            dx = pos[0] - pos_mineral[0]
            dy = pos[1] - pos_mineral[1]
            dist = np.linalg.norm([dx, dy])
            if dist < closest_mineral:
                closest_mineral = dist
        score += 2/(closest_mineral+1)

        return score

    def transition(self, pos, action):
        score = 0
        score += - 0.05
        new_pos = pos + direction_to_delta(action)
        transition = (tuple(pos), tuple(new_pos))
        valid, score_ = check_pos_inside_map(new_pos, self.map_size)
        score += score_
        if valid:
            valid, score_ = check_valid_transition(transition, self.non_valid_transitions)

            score += score_
        if valid:
            score_ = self.get_score(new_pos)
            score += score_

        if not valid:
            new_pos = pos
        return new_pos, score

    def update(self, new_points):
        # generate new hotspots
        if new_points:
            if np.random.random() < self.p_new_hot_spot:
                pos = np.random.randint(self.map_size)
                key = str(tuple(pos))
                self.hot_spots[key] = self.hot_spot_emerge_time

            # increase temperature on new hotspots

            for key in self.hot_spots:
                pos = str_to_tuple(key)
                self.temperature_map[pos] += (1.05/self.temp_decay_factor)*self.mineral_spawn_temp \
                                             / (self.hot_spot_emerge_time-1)

            # map cool down.
            self.temperature_map = self.temp_decay_factor * self.temperature_map
            self.temperature_map = self.temperature_map.clip(self.ambient_temp)

            # if temp high enough generate mineral
            indx = np.argwhere(self.temperature_map > self.mineral_spawn_temp)
            indx = np.fliplr(indx)
            for pos in indx:
                if tuple(pos) not in self.minerals:
                    self.minerals[str(tuple(pos))] = self.generate_mineral(pos)

        # break down old minerals
        keys_to_pop = []
        for key in self.minerals:
            self.minerals[key].time += -1
            if self.minerals[key].time < 0:
                keys_to_pop.append(key)
        for key in keys_to_pop:
            self.minerals.pop(key, None)

        if new_points:
            # reduce time for emerging hot spot.
            keys_to_pop = []
            for key in self.hot_spots:
                self.hot_spots[key] += -1
                if self.hot_spots[key] <= 0:
                    keys_to_pop.append(key)
            for key in keys_to_pop:
                self.hot_spots.pop(key, None)

    def generate_mineral(self, pos):
        copium = 0
        # generate ground composition
        ground_density = np.max([skewnorm.rvs(4, 0.6), 0.4])
        silicon_rate = 0.3*np.random.random()+0.02
        oxygen_rate = 0.1*np.random.random()
        iron_rate = np.random.random() + 0.05
        aluminium_rate = 0.6*np.random.random() + 0.03
        magnesium_rate = 0.25*np.random.random() + 0.01
        undetectable = 0.1*np.random.random() + 0.01
        # normalize
        sum_ = silicon_rate + oxygen_rate + iron_rate + aluminium_rate + magnesium_rate + undetectable
        silicon_rate = silicon_rate/sum_
        oxygen_rate = oxygen_rate/sum_
        iron_rate = iron_rate/sum_
        aluminium_rate = aluminium_rate/sum_
        magnesium_rate = magnesium_rate/sum_
        undetectable = undetectable/sum_

        properties = {'ground_density': ground_density,
                      'moist': np.random.random()*0.3,
                      'reflectivity': np.random.random()*0.6,
                      'silicon_rate': silicon_rate,
                      'oxygen_rate': oxygen_rate,
                      'iron_rate': iron_rate,
                      'aluminium_rate': aluminium_rate,
                      'magnesium_rate': magnesium_rate,
                      'undetectable': undetectable}

        m1 = norm.pdf(properties['ground_density'], 1.2, 0.5) / norm.pdf(1.2, 1.2, 0.5)
        m2 = norm.pdf(properties['moist'], 0.2, 0.1) / norm.pdf(0.2, 0.2, 0.1)
        m3 = norm.pdf(properties['reflectivity'], 0.5, 0.2) / norm.pdf(0.5, 0.5, 0.2)
        m4 = norm.pdf(properties['silicon_rate'], 0.136, 0.15) / norm.pdf(0.136, 0.136, 0.15)
        m5 = norm.pdf(properties['oxygen_rate'], 0.02, 0.1) / norm.pdf(0.02, 0.02, 0.1)
        m6 = norm.pdf(properties['aluminium_rate'], 0.432, 0.1) / norm.pdf(0.432, 0.432, 0.1) + \
             norm.pdf(properties['aluminium_rate'], 0.15, 0.1) / norm.pdf(0.15, 0.15, 0.1)
        m = m1*m2*m3+m4*m5*m6

        if m > 0.1 and m < 0.7:
            copium = 1
        return Mineral(properties=properties, mineral=copium)

    def sample(self, pos):
        key = str(tuple(pos))
        if key in self.minerals:
            properties = self.minerals[key].properties
            mineral = self.minerals[key].mineral
            self.minerals.pop(key, None)
            return properties, mineral
        else:
            return None, None


class Mineral(object):
    def __init__(self, properties, mineral):
        self.time = 20
        self.properties = properties
        self.mineral = mineral


def direction_to_delta(direction):
    '''
    :param direction: 'N', 'S', 'E', 'W'
    :return: np.array([dx, dy])
    '''
    x = 0
    y = 0
    if direction == 'N':
        y = -1
    elif direction == 'S':
        y = 1
    elif direction == 'E':
        x = 1
    elif direction == 'W':
        x = -1

    return np.array([x, y])


def check_pos_inside_map(pos, map_size):
    fail_score = -5
    if np.min(pos) < 0:
        return False, fail_score

    if pos[0] >= map_size[0] or pos[1] >= map_size[1]:
        return False, fail_score

    return True, 0


def check_valid_transition(transition, non_valid_transitions):
    if transition in non_valid_transitions:
        return False, -5
    return True, 0


def str_to_tuple(string_pos):
    # remove parentheses
    s1 = string_pos.split('(')
    s2 = s1[1].split(')')
    s3 = s2[0]
    # split comma
    s4 = s3.split(',')
    # map to int
    s5 = map(int, s4)
    # create tuple
    return tuple(s5)

class PLT_ACC(object):
    def __init__(self, renderer, tile_res_w, tile_res_h):
        self.renderer = renderer
        self.accuracy_tree = 0
        self.accuracy_knn = 0

        self.tile_res_w = tile_res_w
        self.tile_res_h = tile_res_h
        self.my_font = pygame.font.SysFont('Comic Sans MS', 20)

        surf = pygame.Surface((int(5 * self.tile_res_w), int(8 * self.tile_res_h)), pygame.SRCALPHA)
        self.obj = pygame_render.RenderObject(surf=surf, x=int(22.6*tile_res_w), y=int(21.1 * tile_res_h))
        self.renderer.add_object(key='plot_acc', layer='plot', obj=self.obj)
        self.update_acc(0, 0)

    def update_acc(self, accuracy_tree, accuracy_knn):
        self.accuracy_tree = accuracy_tree
        self.accuracy_knn = accuracy_knn

        surf = pygame.Surface((int(5*self.tile_res_w), int(8*self.tile_res_h)), pygame.SRCALPHA)



        rect = pygame.Rect(0, 0, int(5*self.tile_res_w), int(8*self.tile_res_w))
        pygame.draw.rect(surface=surf, color=(180, 0, 0, 70), rect=rect)

        knn_h = int(self.accuracy_knn*6 * self.tile_res_w)
        rect_kkn = pygame.Rect(int(3.5 * self.tile_res_w), int(7 * self.tile_res_h)- knn_h, int(1 * self.tile_res_w), knn_h)
        pygame.draw.rect(surface=surf, color=(0, 255, 255, 200), rect=rect_kkn)

        tree_h = int(self.accuracy_tree*6 * self.tile_res_w)
        rect_tree = pygame.Rect(int(1.5 * self.tile_res_w), int((7) * self.tile_res_h) - tree_h, int(1 * self.tile_res_w), tree_h)
        pygame.draw.rect(surface=surf, color=(0, 255, 255, 200), rect=rect_tree)

        color =(255, 255, 255, 255)
        pygame.draw.line(surface=surf, color=color, start_pos=(int(self.tile_res_w), int(self.tile_res_h)),
                         end_pos=(int(self.tile_res_w), int(7*self.tile_res_h)), width=3)
        pygame.draw.line(surface=surf, color=color, start_pos=(int(self.tile_res_w), int(7*self.tile_res_h)),
                                                               end_pos=(int(5*self.tile_res_w), int(7*self.tile_res_h)), width=2)
        pygame.draw.line(surface=surf, color=color, start_pos=(int(0.8*self.tile_res_w), int(self.tile_res_h)),
                         end_pos=(int(1.2*self.tile_res_w), int(self.tile_res_h)), width=2)

        knn_text = self.my_font.render("knn", False, (255, 255, 255))
        surf.blit(knn_text, (int(3.5 * self.tile_res_w), int(7 * self.tile_res_h)))

        tree_text = self.my_font.render("tree", False, (255, 255, 255))
        surf.blit(tree_text, (int(1.5 * self.tile_res_w), int(7 * self.tile_res_h)))

        acc_text = self.my_font.render("Accuracy", False, (255, 255, 255))
        surf.blit(acc_text, (int(1.5 * self.tile_res_w), int(0* self.tile_res_h)))

        self.obj.update_surface(surf)


def load_backdrop(map_type, renderer):
    if map_type == 1:
        img = pygame.image.load("imgs/ground6x6_v2.png").convert()
    else:
        img = pygame.image.load("imgs/ground25x25.png").convert()
    w, h = renderer.get_resolution()
    surf = pygame.Surface((w, h))
    surf.blit(img, (0,0))
    obj = pygame_render.RenderObject(surf=surf, x=int(w/2), y=int(h/2))
    renderer.add_object(key='back_ground', layer='backdrop', obj=obj)