from functions import k_means, calculate_accuracy

df = k_means(
    number_of_cluster=6,
    x="Temperature",
    y="L",
    file_dir='./Project2/Stars.csv',
    visualization_colors=["blue", "yellow", "green", "brown", "purple", "orange"]
)

accuracy = calculate_accuracy(df, 6)
print(f"accuracy of clustering = {accuracy}%")
