import pygame
import sys
import dbscan

pygame.init()
eps = 30
min_pts = 3

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))

points = []
points_colors = []
default_color = "black"
colors = ['red', 'green', 'purple', 'yellow', 'blue']

running = True
while running:
    screen.fill("white")

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                points.append(event.pos)
                points_colors.append(default_color)

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                points = []
                points_colors = []
            elif event.key == pygame.K_RETURN:
                cluster_labels = dbscan.dbscan_algorithm(points, eps, min_pts)
                for i, label in enumerate(cluster_labels):
                    if label == -1:
                        points_colors[i] = "black"
                    else:
                        points_colors[i] = colors[label % len(colors)]

    for i, point in enumerate(points):
        pygame.draw.circle(screen, points_colors[i], point, 5)

    pygame.display.flip()

pygame.quit()
sys.exit()
