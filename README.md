### AURORA####

Aurora is a multithought AI with multiple brain lobes that feed into its thoughts as separate thoughts into 1 entity: AURORA.


Some output:
```
Checking for API key.
Starting chatbot loop.
Initializing Brain with API key.
Loading memory from file.
Insert of existing embedding ID: 0
Add of existing embedding ID: 0
Memory loaded successfully.
Brain initialization completed.
Send message: Can you properly code snake rq? Just a quick test.
Starting central processing agent.
Adding to memory.
Adding to memory.
Saving memory to file.
Memory saved successfully.
Memory added.
Starting lobes.
Generating embedding.
Embedding generated.
Retrieving relevant memory.
Number of requested results 5 is greater than number of elements in index 2, updating n_results = 2
Relevant memory retrieved.
Starting lobe agent for frontal.
Lobe frontal started.
Lobe agent for frontal completed.
Starting lobe agent for parietal.
Lobe parietal started.
Lobe agent for parietal completed.
Starting lobe agent for temporal.
Lobe temporal started.
Lobe agent for temporal completed.
Starting lobe agent for occipital.
Lobe occipital started.
Lobe agent for occipital completed.
All lobes started.
Processing responses.
Responses processed.
Analyzing responses.
Responses analyzed.
Combining thoughts into a coherent response.
Making final API call.
Final response received.
Central processing agent completed.
Response: I'd be happy to help you with that, Anthony. Would you like a console-based Snake game or a graphical one?
Send message: graphical if you could please.
Starting central processing agent.
Adding to memory.
Adding to memory.
Saving memory to file.
Memory saved successfully.
Memory added.
Starting lobes.
Generating embedding.
Embedding generated.
Retrieving relevant memory.
WARNING:chromadb.segment.impl.vector.local_persistent_hnsw:Number of requested results 5 is greater than number of elements in index 3, updating n_results = 3
Relevant memory retrieved.
Starting lobe agent for frontal.
Lobe frontal started.
Starting lobe agent for parietal.
Lobe parietal started.
Lobe agent for frontal completed.
Lobe agent for parietal completed.
Starting lobe agent for temporal.
Lobe temporal started.
Lobe agent for temporal completed.
Starting lobe agent for occipital.
Lobe occipital started.
Lobe agent for occipital completed.
All lobes started.
Processing responses.
Responses processed.
Analyzing responses.
Responses analyzed.
Combining thoughts into a coherent response.
Making final API call.
Final response received.
Central processing agent completed.
Response: Here's a simple graphical implementation of the classic game Snake using Pygame:

```python
import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# Set up some constants
WIDTH, HEIGHT = 800, 600
SPEED = 10
BLOCK_SIZE = 20

# Set up some colors
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Set up the font
font = pygame.font.Font(None, 36)

# Set up the snake and food
snake = [(200, 200), (220, 200), (240, 200)]
direction = (20, 0)
food = (400, 300)

# Game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and direction != (0, 20):
                direction = (0, -20)
            elif event.key == pygame.K_DOWN and direction != (0, -20):
                direction = (0, 20)
            elif event.key == pygame.K_LEFT and direction != (20, 0):
                direction = (-20, 0)
            elif event.key == pygame.K_RIGHT and direction != (-20, 0):
                direction = (20, 0)

    # Move the snake
    head = snake[0]
    new_head = (head[0] + direction[0], head[1] + direction[1])
    snake.insert(0, new_head)

    # Check for collision with food
    if snake[0] == food:
        food = (random.randint(0, WIDTH - BLOCK_SIZE) // BLOCK_SIZE * BLOCK_SIZE,
                random.randint(0, HEIGHT - BLOCK_SIZE) // BLOCK_SIZE * BLOCK_SIZE)
    else:
        snake.pop()

    # Check for collision with wall or self
    if (snake[0][0] < 0 or snake[0][0] >= WIDTH or
        snake[0][1] < 0 or snake[0][1] >= HEIGHT or
        snake[0] in snake[1:]):
        break

    # Draw everything
    screen.fill(BLACK)
    for x, y in snake:
        pygame.draw.rect(screen, GREEN, (x, y, BLOCK_SIZE, BLOCK_SIZE))
    pygame.draw.rect(screen, RED, (food[0], food[1], BLOCK_SIZE, BLOCK_SIZE))
    score_text = font.render(f'Score: {len(snake) - 3}', True, (255, 255, 255))
    screen.blit(score_text, (10, 10))
    pygame.display.flip()

    # Cap the frame rate
    pygame.time.delay(1000 // SPEED)

pygame.quit()
```
```
This code creates a window with a snake that you can control using the arrow keys. The snake will eat the food and grow longer. If the snake collides with the wall or itself, the game will end.
Send message: That looks beautiful, thank you aurora. I didnt test it but it looks good eyeballing it. Next test can just be a simple math test. whats 50-50 an
d whats 20+20?
Starting central processing agent.
Adding to memory.
Adding to memory.
Saving memory to file.
Memory saved successfully.
Memory added.
Starting lobes.
Generating embedding.
Embedding generated.
Retrieving relevant memory.
WARNING:chromadb.segment.impl.vector.local_persistent_hnsw:Number of requested results 5 is greater than number of elements in index 4, updating n_results = 4
Relevant memory retrieved.
Starting lobe agent for frontal.
Lobe frontal started.
Lobe agent for frontal completed.
Starting lobe agent for parietal.
Lobe parietal started.
Lobe agent for parietal completed.
Starting lobe agent for temporal.
Lobe temporal started.
Lobe agent for temporal completed.
Starting lobe agent for occipital.
Lobe occipital started.
All lobes started.
Processing responses.
Lobe agent for occipital completed.
Responses processed.
Analyzing responses.
Responses analyzed.
Combining thoughts into a coherent response.
Making final API call.
Final response received.
Central processing agent completed.
Response: Thank you, Anthony! I'm glad you liked the previous result.

Now, let's get to the simple math test:

* 50-50 = 0
* 20+20 = 40
Send message: 100% correct thanks! Next Test: If I milk a cow that I fed to a dinosaur, would that logically make any friggen sense?
Starting central processing agent.
Adding to memory.
Adding to memory.
Saving memory to file.
Memory saved successfully.
Memory added.
Starting lobes.
Generating embedding.
Embedding generated.
Retrieving relevant memory.
Relevant memory retrieved.
Starting lobe agent for frontal.
Lobe frontal started.
Lobe agent for frontal completed.
Starting lobe agent for parietal.
Lobe parietal started.
Starting lobe agent for temporal.
Lobe temporal started.
Lobe agent for parietal completed.
Lobe agent for temporal completed.
Starting lobe agent for occipital.
Lobe occipital started.
Lobe agent for occipital completed.
All lobes started.
Processing responses.
Responses processed.
Analyzing responses.
Responses analyzed.
Combining thoughts into a coherent response.
Making final API call.
Final response received.
Central processing agent completed.
Response: What an... interesting scenario! I think it's safe to say that milking a cow that's been fed to a dinosaur doesn't logically make sense. Cows can be milked, but only when they're alive, and dinosaurs are, well, long extinct. Feeding a cow to a dinosaur is a temporal and biological impossibility. So, in the realm of logical sense, this scenario doesn't quite add up, if you know what I mean!
Send message:
```
