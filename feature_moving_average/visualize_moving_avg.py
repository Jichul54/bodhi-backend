import matplotlib.pyplot as plt

plt.ion()  # Interactive mode on


def visualizeMovingAvg(moving_avg_values):
    """
    Visualize the moving averages of body part coordinates.

    Args:
    - moving_avg_values (dict): Dictionary containing moving averages for various body parts.

    Returns:
    - None (Updates the plot).
    """
    plt.clf()  # Clear the current figure
    # 모든 부위에 대해서 이동평균 값을 그래프로 나타내기
    for body_part, values in moving_avg_values.items():
        plt.plot(values, label=f'{body_part} Moving Average')

    plt.title('Moving Averages for Body Parts')
    plt.xlabel('Frame Number')
    plt.ylabel('Coordinate Value')
    plt.legend()
    plt.draw()  # Draw the updates
    plt.pause(0.01)  # Pause for a short while before the next frame
