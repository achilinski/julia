using JLD2

# Load both variables
data = jldopen("imdb_dataset_prepared.jld2", "r") do file
    (
        reviews = file["reviews"],
        labels = file["labels"]
    )
end

# Show the first 10 rows of each
println("First 2 reviews:")
for i in 1:2
    println(data.reviews[i])
end

println("\nFirst 2 labels:")
for i in 1:2
    println(data.labels[i])
end
