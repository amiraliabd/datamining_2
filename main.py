from functions import k_means, calculate_accuracy, standard_kmeans


df = k_means(
    number_of_cluster=6,
    x="Temperature",
    y="L",
    file_dir='./Project2/Stars.csv',
    visualization_colors=["blue", "yellow", "green", "brown", "purple", "orange"]
)

accuracy = calculate_accuracy(df, 6)
print(f"accuracy of clustering = {accuracy}%")

dataset_dir = './Project2/dataset.csv'
standard_kmeans(
    dataset_dir=dataset_dir,
    k=3,
    ds_row=1,
    ds_column=20,
    x='absence possibility',
    y='Month of absence'
)

