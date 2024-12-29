import pygame
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# Initialize Pygame
pygame.init()

# Set up display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Curvilinear Triangle with Shapely")

# Define colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)

# Define points for the triangle
p1 = Point(300, 500)
p2 = Point(400, 100)
p3 = Point(500, 500)

# Create a polygon (curvilinear triangle) using the points
triangle = Polygon([p1, p2, p3])

# Function to create Bezier curves for the triangle edges
def bezier_curve(p0, p1, p2, num_points=100):
    points = []
    for t in np.linspace(0, 1, num_points):
        x = (1 - t) ** 2 * p0.x + 2 * (1 - t) * t * p1.x + t ** 2 * p2.x
        y = (1 - t) ** 2 * p0.y + 2 * (1 - t) * t * p1.y + t ** 2 * p2.y
        points.append((x, y))
    return points

# Control points for the curves
control1 = Point(350, 150)  # Control point for the curve between p1 and p2
control2 = Point(450, 150)  # Control point for the curve between p2 and p3

# Create the curves for the triangle edges
curve1 = bezier_curve(p1, control1, p2)  # Curve from p1 to p2
curve2 = bezier_curve(p2, control2, p3)  # Curve from p2 to p3

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Clear the screen
    screen.fill(WHITE)

    # Draw the curves for the sides
    pygame.draw.aalines(screen, BLUE, False, curve1)
    pygame.draw.aalines(screen, BLUE, False, curve2)

    # Draw the base as a straight line
    pygame.draw.line(screen, BLUE, (int(p1.x), int(p1.y)), (int(p3.x), int(p3.y)), 1)

    # Draw vertices
    pygame.draw.circle(screen, BLUE, (int(p1.x), int(p1.y)), 5)
    pygame.draw.circle(screen, BLUE, (int(p2.x), int(p2.y)), 5)
    pygame.draw.circle(screen, BLUE, (int(p3.x), int(p3.y)), 5)

    # Update display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
