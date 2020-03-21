from sklearn.cluster import KMeans
import numpy as np
#2d model
import matplotlib.pyplot as plt

# words count, reviews count, year
data = np.array([ 
					[10201,34158,1943],
					[95022,43814,1937],
					[36363,19427,1950],
					[73404,52908,1951],
					[47094,61698,1925],
					[144739,27157,1967],
					[112473,22391,1955],
					[99121,19331,1908],
					[136090,6693,1996],
					[46118,42764,1953],
					[69066,9016,1876],
					[155960,6807,1839],
					[206052,14082,1851],
					[349736,21743,1878],
					[364153,9712,1879],
					[78462,26138,1890],
					[63766,26700,1932],
					[86275,26128,1947],
					[88942,61203,1949],
					[36830,20156,1942],
					[28710,13400,1864],
					[99760,59169,1813],
					[59000,22630,1897],
					[52345,27474,1939],
					[76944, 99202,1997],
					[106865,85011,1960],
					[275500,6186,1922],
					[22185,14835,1915],
					[21460,22181,1952],
					[97295,164728,2008],
					[124265,106883,2005],
					[144855,45071,2003],
					[181540,36112,2001],
					[111795,99414,2005],
					[95265,22616,2005],
					[109185,23918,1990],
					[177227,19790,1954],
					[220835,17653,1952],
					[112085,45872,2001],
])

#titles of books
labels = np.array([
					"The Little Prince", 
					"The Hobbit",
					"The Lion, the Witch and the Wardrobe",
					"The Catcher in the Rye",
					"The Great Gatsby",
					"One Hundred Years of Solitude",
					"Lolita",
					"Anne of Green Gables",
					"The Green Mile",
					"Fahrenheit 451",
					"The Adventures of Tom Sawyer",
					"Oliver Twist",
					"Moby Dick",
					"Anna Karenina",
					"The Brothers Karamazov",
					"The Picture of Dorian Gray",
					"Brave New World",
					"The Diary of a Young Girl",
					"Nineteen Eighty-Four",
					"The Stranger",
					"Alice's Adventures in Wonderland",
					"Pride and Prejudice",
					"Dracula",
					"Ten Little Niggers",
					"Harry Potter and the Sorcerer's Stone",
					"To Kill a Mockingbird",
					"Ulysses",
					"The Metamorphosis",
					"The Old Man and the Sea",
					"The Hunger Games",
					"The Book Thief",
					"The Da Vinci Code",
					"American Gods",
					"Twilight",
					"Extremely Loud and Incredibly Close",
					"Good Omens",
					"The Lord of the Rings - The Fellowship of the Ring",
					"East of Eden",
					"Life of Pi"
					])

N_CLUSTERS = 5

kmeans = KMeans(init='k-means++', n_clusters=N_CLUSTERS, n_init=10)
kmeans.fit(data)
pred_classes = kmeans.predict(data)

centroids = kmeans.cluster_centers_
print(centroids)

for cluster in range(N_CLUSTERS):
    print('cluster: ', cluster)
    print(labels[np.where(pred_classes == cluster)])
  