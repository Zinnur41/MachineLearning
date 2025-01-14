import pygame
import sys
import dbscan

pygame.init()

# DBSCAN Parameters
eps = 30
min_pts = 3

# Screen Dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Points and Colors
points = []
points_colors = []
default_color = "black"
colors = ['red', 'green', 'purple', 'yellow', 'blue']

# Mouse state
mouse_held = False

running = True
while running:
    screen.fill("white")

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                mouse_held = True
                points.append(event.pos)
                points_colors.append(default_color)

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button
                mouse_held = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:  # Clear points
                points = []
                points_colors = []
            elif event.key == pygame.K_RETURN:  # Perform DBSCAN
                cluster_labels = dbscan.dbscan_algorithm(points, eps, min_pts)
                for i, label in enumerate(cluster_labels):
                    if label == -1:
                        points_colors[i] = "black"
                    else:
                        points_colors[i] = colors[label % len(colors)]

    # Add points while the mouse is held down
    if mouse_held:
        pos = pygame.mouse.get_pos()
        if not points or pos != points[-1]:  # Avoid adding duplicate points
            points.append(pos)
            points_colors.append(default_color)

    # Draw points
    for i, point in enumerate(points):
        pygame.draw.circle(screen, points_colors[i], point, 5)

    pygame.display.flip()

pygame.quit()
sys.exit()
