import pygame
import neat
import time
import math
import os
import random
import copy
import numpy as np

import Bird 
import Pipe
import Terrain

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 800

pygame.font.init()
STAT_FONT = pygame.font.SysFont("comicsans",50)

shouldExit = False

def draw_window(win, birds, pipes, base, score):
    win.blit(Terrain.BG_IMG, (0,0))

    for pipe in pipes :
        pipe.draw(win)

    for bird in birds :
        bird.draw(win)
    
    text = STAT_FONT.render("Score: "+str(score),1,(255,255,255))
    win.blit(text, (WINDOW_WIDTH-10-text.get_width(),10))

    base.draw(win)
    pygame.display.update()

def main(genomes, config):
    global shouldExit

    if shouldExit:
        exit(0)

    nets = []
    ge = []
    traj = []
    birds = []

    dnets = []
    dge = []
    dtraj = []
    dbirds = []

    for _,g in genomes:
        # print(list(g.nodes.keys()), list(g.connections.keys()), list(map(lambda x: x.weight, g.connections.values())))
        net = neat.nn.FeedForwardNetwork.create(g,config)
        nets.append(net)
        birds.append(Bird.Bird(230,350))
        g.fitness = 0
        ge.append(g)
        traj.append(1)

    base = Terrain.Base(730)
    pipes = [Pipe.Pipe(700)]
    win = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    score = 0

    run = True
    while run :
        clock.tick(30)
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        # bird move
        pipe_ind = 0
        if len(birds)>0:
            if len(pipes)>1 and birds[0].x>pipes[0].x+pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            run = False
            break

        for x, bird in enumerate(birds):
            bird.move()
            # ge[x].fitness += 0.3

            output = nets[x].activate((bird.y,abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))
            if output[0] >0.5 :
                bird.jump()
                traj[x] = traj[x]*10+1
            else:
                traj[x] += traj[x]*10
        
        add_pipe = False
        rem = []
        for pipe in pipes:
            for x,bird in enumerate(birds) :
                if pipe.collide(bird):
                    # ge[x].fitness -= 0.5
                    dbirds.append(birds.pop(x))
                    dnets.append(nets.pop(x))
                    dge.append(ge.pop(x))
                    dtraj.append(traj.pop(x))

                if not pipe.passed and pipe.x<bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width()<0:
                rem.append(pipe)
            pipe.move()

        if add_pipe :
            score += 1
            # for g in ge :
            #     g.fitness += 5
            pipes.append(Pipe.Pipe(600))

        for r in rem :
            pipes.remove(r)
        
        for x,bird in enumerate(birds) :
            if bird.y + bird.img.get_height() >= 730 or bird.y<0 :
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        base.move()
        draw_window(win, birds, pipes, base, score)

    padlen = max(list(map(lambda x: len(str(x)), dtraj)))
    for i in range(len(dtraj)):
        dtraj[i] /= 10**padlen

    fitnesses = []
    for i in range(len(dtraj)):
        distances = list(map(lambda x: abs(x-dtraj[i]), dtraj))
        fitnesses.append(novelty(distances))
        dge[i].fitness = fitnesses[-1]

    variance = lambda data, avg: sum([x**2 for x in [i-avg for i in data]])/float(len(data))
    mean = lambda data: float(sum(data)/len(data))
    std_dev = lambda data: math.sqrt(variance(data, mean(data)))

    if max(fitnesses)>(mean(fitnesses)+5*(std_dev(fitnesses))):
        shouldExit = True

def novelty(distances):
    idx = np.argsort(distances)

    mean = 0
    for i in idx[1:9]:
        mean += distances[i]
    mean /= 10

    return mean

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, \
            neat.DefaultReproduction, \
            neat.DefaultSpeciesSet, \
            neat.DefaultStagnation, \
            config_path )
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main,50)

if __name__=='__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
