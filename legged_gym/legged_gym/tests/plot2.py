import matplotlib.pyplot as plt

# Sample data similar to the graph in the image
hurdle_heights = [0.24, 0.26, 0.30, 0.35, 0.40]
success_rates_1 = [1.0, 0.9, 0.6, 0.3, 0.1]
success_rates_2 = [1.0, 0.85, 0.65, 0.5, 0.2]
success_rates_3 = [1.0, 0.8, 0.5, 0.4, 0.3]
success_rates_4 = [1.0, 0.9, 0.6, 0.3, 0.1]
success_rates_5 = [1.0, 0.85, 0.65, 0.5, 0.2]
success_rates_6 = [1.0, 0.8, 0.5, 0.4, 0.3]
success_rates_7 = [1.0, 0.9, 0.6, 0.3, 0.1]
success_rates_8 = [1.0, 0.85, 0.65, 0.5, 0.2]
success_rates_9 = [1.0, 0.8, 0.5, 0.4, 0.3]

# Plot each series with a different color and marker
plt.plot(hurdle_heights, success_rates_1, 'b-s', label='Series 1')  # Blue square markers
plt.plot(hurdle_heights, success_rates_2, 'r-^', label='Series 2')  # Red triangle markers
plt.plot(hurdle_heights, success_rates_3, 'g-o', label='Series 3')  # Green circle markers

# Adding the dashed vertical line at x = 0.26, similar to the image
plt.axvline(x=0.26, color='orange', linestyle='--')

# Add legend, grid, labels, and title
plt.legend()
plt.grid(True)
plt.xlabel('Hurdle Height [m]')
plt.ylabel('Success Rate [%]')
plt.title('Success Rate by Hurdle Height')

# Show the plot
plt.show()
