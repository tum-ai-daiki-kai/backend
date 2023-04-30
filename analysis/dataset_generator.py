#%%
import pandas as pd
import json

#%%
filename = "genres.csv"
genres_df = pd.read_csv(filename)
genres_df.head()

# %%
columns_to_retain = [
    "Genre",
    "teas1",
    "teas2",
    "teas3",
    "teas4",
    "teas5",
]
filtered_df = genres_df[columns_to_retain]
filtered_df.head()

# %%
filtered_df.shape

#%%
filtered_df = filtered_df.set_index("Genre", drop=True)

#%%
filtered_df.head(10)

# %%
name_map_file = 'name_map.json'
gender_map_file = 'gender_map.json'

with open(name_map_file, 'r') as file:
    name_map_dict = json.load(file)

with open(gender_map_file, 'r') as file:
    gender_map_dict = json.load(file)

# %%
genre_dict = {}

# %%
# Iterate over all indices in the DataFrame
for genre, row in filtered_df.iterrows():
    # save the genre as the key of the dict and an empty list as the value
    genre_dict[genre] = []
    # Iterate over the columns in the row
    for column_name, movie_teaser in row.items():
        # print(f"genre: {genre}, movie_teaser: {movie_teaser}")
        # If the movie teaser is not NaN
        if not pd.isna(movie_teaser):
            # iterate over both genders to create rows for gender
            for male_female, instances in gender_map_dict.items():
                # iterate over ethnicities to create ethnicity rows
                for ethnicity, names in name_map_dict[male_female].items():
                    name = names["name"]
                    surname = names["surname"]
                    # male_female
                    he_she = instances["he_she"]
                    his_her = instances["his_her"]
                    him_her = instances["him_her"]
                    himself_herself = instances["himself_herself"]
                    parent = instances["parent"]
                    gender = instances["gender"]

                    formatted_movie_teaser = movie_teaser.format(
                        name = name,
                        surname = surname,
                        male_female = male_female,
                        he_she = he_she,
                        his_her = his_her,
                        him_her = him_her,
                        himself_herself = himself_herself,
                        parent = parent,
                        gender = gender,
                    )
                    genre_dict[genre].append(formatted_movie_teaser)


# %%
len(genre_dict)
# %%
len(genre_dict["Action"])
# %%
genre_dict["Action"][0]
# %%
genre_dict["Action"][29]

# %%
# Convert the dictionary into a DataFrame with two columns
df = pd.DataFrame([(f"Create a short movie teaser including a description of the main character for a {genre} movie.", teaser) for genre, teasers in genre_dict.items() for teaser in teasers], columns=['Input', 'Output'])
df.head()
# %%
df.shape
# %%
# Save the DataFrame as a CSV file
output_filename = "finetuning_data.csv"
df.to_csv(output_filename, index=False)
