import pygame
import random
import os
import time
import neat


# Define Variables
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
BIRD_IMGS = [
    pygame.image.load(os.path.join("assets/sprites/yellowbird-downflap.png")),
    pygame.image.load(os.path.join("assets/sprites/yellowbird-midflap.png")),
    pygame.image.load(os.path.join("assets/sprites/yellowbird-upflap.png"))
]
# 512 / 361 so 48 / 33.75
for i in range(len(BIRD_IMGS)):
    BIRD_IMGS[i] = pygame.transform.scale(BIRD_IMGS[i], (48, 33.75))


PIPE_IMG = pygame.image.load(os.path.join("assets/sprites/pipe-green-down.png"))
# 163 / 1000 then 163 /2 = 81.5
PIPE_IMG = pygame.transform.scale(PIPE_IMG, (163/2, 1000/2))

BASE_IMG = pygame.image.load(os.path.join("assets/sprites/base.png"))
BASE_IMG = pygame.transform.scale(BASE_IMG, (336, 112))
BG_IMG = []
BG_N_REPEAT = WINDOW_WIDTH // 288 + 1
for i in range(BG_N_REPEAT):
    BG_IMG.append(pygame.image.load(os.path.join("assets/sprites/background-day.png")))

PIPE_H_GAP = 200
PIPE_V_GAP = 200

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Flappy Bird")
pygame.font.init()
FONT = pygame.font.SysFont("comicsans", 20)

GENERATION = 0
# Define classes

class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0 # degrees
        self.tickCount = 0
        self.vel = 0
        self.height = self.y
        self.imgCount = 0
        self.img = self.IMGS[1]
    
    def jump(self):
        self.vel = -10.5
        self.tickCount = 0
        self.height = self.y
    
    def move(self):
        self.tickCount += 1
        displacement = self.vel * self.tickCount + 1.5 * self.tickCount**2
        if displacement >= 16:
            displacement = 16
        if displacement < 0:
            displacement -= 2
        self.y += displacement
        if displacement < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL
    
    def draw(self):
        self.imgCount += 1
        self.img = self.IMGS[self.imgCount // self.ANIMATION_TIME % len(self.IMGS)]
        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.imgCount = self.ANIMATION_TIME * 2        
        rotated_img = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_img.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        screen.blit(rotated_img, new_rect.topleft)
    
    def get_mask(self):
        return pygame.mask.from_surface(self.img)

class Pipe:
    VEL = 5
    BASE_H = 112
    PIPE_MIN_H = round(75/ 2)
    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG
        self.passed = False
        self.setHeight()
    
    def setHeight(self):
        self.height = random.randrange(self.PIPE_MIN_H, WINDOW_HEIGHT - self.BASE_H - self.PIPE_MIN_H - PIPE_V_GAP)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + PIPE_V_GAP
    
    def move(self):
        self.x -= self.VEL
    
    def draw(self):
        screen.blit(self.PIPE_TOP, (self.x, self.top))
        screen.blit(self.PIPE_BOTTOM, (self.x, self.bottom))
    
    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))
        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)
        if b_point or t_point:
            return True
        return False

class Base:
    VEL = 5
    WIDTH = BASE_IMG.get_width() # 336
    IMG = BASE_IMG
    def __init__(self, y):
        self.y = y
        self.n = WINDOW_WIDTH // self.WIDTH + 2
        self.bases = []
        for i in range(self.n):
            self.bases.append((self.WIDTH * i, self.y))
    
    def move(self):
        for i in range(self.n):
            self.bases[i] = (self.bases[i][0] - self.VEL, self.bases[i][1])
        if self.bases[0][0] + self.WIDTH <= 0:
            self.bases.pop(0)
            self.bases.append((self.bases[-1][0] + self.WIDTH, self.y))

    def draw(self):
        for i in range(self.n):
            screen.blit(self.IMG, self.bases[i])


    


def eval_bird(genomes, config):
    global GENERATION
    GENERATION += 1
    birds = []
    ge = []
    nets = []
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(WINDOW_WIDTH // 2 - BIRD_IMGS[0].get_width() // 2, WINDOW_HEIGHT // 2 - BIRD_IMGS[0].get_height() // 2))
        g.fitness = 0
        ge.append(g)
    
    score = 0
    base = Base(WINDOW_HEIGHT - 112)
    pipes = [Pipe(WINDOW_WIDTH - PIPE_H_GAP)]

    run = True
    clock = pygame.time.Clock()
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_ESCAPE] or pressed[pygame.K_q]:
            run = False
            pygame.quit()
            quit()

        pipe_ind = 0
        if len(birds) > 0:
           for x, pipe in enumerate(pipes):
                if pipe.x + pipe.PIPE_TOP.get_width() > birds[0].x:
                    pipe_ind = x
                    break
        else:
            run = False
            break


        for x, bird in enumerate(birds):
            # ge[x].fitness += 0.05
            bird.move()
            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))
            if output[0] > 0.5:
                bird.jump()

        base.move()
        
        pipes_to_remove = []
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)
                    continue

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    score += 1
                    ge[x].fitness += 5
                    # pipes_to_remove.append(pipe)

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                pipes_to_remove.append(pipe)
            
            pipe.move()
        for pipe in pipes_to_remove:
            pipes.remove(pipe)
        if pipes[-1].x < WINDOW_WIDTH - PIPE_H_GAP:
            pipes.append(Pipe(WINDOW_WIDTH))


        for x, bird in enumerate(birds):
            if(bird.y + bird.img.get_height() >= WINDOW_HEIGHT - 112) or bird.y < 0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        screen.fill((0x4d, 0xc1, 0xcb))
        for i in range(BG_N_REPEAT):
            screen.blit(BG_IMG[i], (288 * i, WINDOW_HEIGHT - 512))
        for pipe in pipes:
            pipe.draw()
        for bird in birds:
            bird.draw()
        base.draw()
        text = FONT.render(f"Score: {score}", 1, (255, 255, 255))
        screen.blit(text, (10 , 10))
        text = FONT.render(f"Generation: {GENERATION}", 1, (255, 255, 255))
        screen.blit(text, (10, 10 + text.get_height()))
        text = FONT.render(f"Birds Alive: {len(birds)}", 1, (255, 255, 255))
        screen.blit(text, (10, 10 + text.get_height() * 2))
        pygame.display.update()
    
    




def run(config_file):
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_file)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # 50 generations
    winner = p.run(eval_bird, 50)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "NEAT-config-feedforward.txt")
    run(config_path)