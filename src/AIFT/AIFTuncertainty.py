import csv
import numpy as np
from operator import itemgetter
import src.SAL.settings as settings

# output file
f1 = open("original.csv", "w+")
f2 = open("uncertain.csv", "w+")

# storage for the 15 patches probability vector
p = [[0, 0]]
for x in range(14):
	p.append([0, 0])

# used to calculate a to decide top/bottom α
p1 = [0]
for x in range(14):
	p1.append(0)
	
p2 = [0]
for x in range(14):
	p2.append(0)
	
# storage for the final uncertain value
uncertainty = [0]
for x in range(14):
	uncertainty.append(0)

# set lamda variables here
lamda1 = 0.5
lamda2 = 0.5

# retrieve the 15 probabilities and store for each candidate
with open(settings.search_feature) as csvfile:  # settings.search_feature --> 'search_prob_caltech-256.csv'
	readCSV = csv.reader(csvfile, delimiter=',')
	n = 0
	full = 0

	for row in readCSV:
		# this will ignore the original instance of the image (we only need the patch's probabilities to create the vector)
		if n == 0:
			n = n + 1  # do nothing
			
		else:
			if n == 15:
				p[n-1][0] = float(row[0])
				p[n-1][1] = float(row[1])
				p1[n-1] = float(row[0])
				p2[n-1] = float(row[1])
				n = 0
				full = 1  # indicates that all the data for one candidate has been retrieved
			else:
				p[n-1][0] = float(row[0])
				p[n-1][1] = float(row[1])
				p1[n-1] = float(row[0])
				p2[n-1] = float(row[1])
				n = n + 1
			
		# now that we have the probability vectors
		if full == 1:
			# sort it and turn it into a numpy array
			p = sorted(p, key=itemgetter(0))
			p = np.array(p)
			
			p1 = np.array(p1)
			p2 = np.array(p2)
			
			# compute a to determine if we are going to use top or bottom α, since α is 1/4, we'll use 4 out of the 15
			a = np.sum(p1) / 15
			
			if a > 0.5:
				# use top α, which would be in the bottom since its sorted in ascending order
				arr = p[11:]
			else:
				# use bottom α
				arr = p[0:4]
			
			# now that we have the ones we will be using, we can compute the R matrix (4x4)
			w, h = 4, 4
			R = [[0 for x in range(w)] for y in range(h)]
			sum = 0
			error = 1.0e-60
			for i in range(4):
				for j in range(4):
					if i == j:
						# print("do entropy here");
						R[i][j] = -(((arr[i][0]) * np.log(error+arr[i][0])) + ((arr[i][1]) * np.log(error+arr[i][1])))
						R[i][j] = lamda1 * R[i][j]  # apply lamda here
						sum += R[i][j]
					else:
						R[i][j] = ((arr[i][0] - arr[j][0]) * np.log((error+arr[i][0]) / (error+arr[j][0]))) + ((arr[i][1] - arr[j][1]) * np.log((error+arr[i][1]) / (error+arr[j][1])));
						R[i][j] = lamda2 * R[i][j]  # apply lamda here
						sum += R[i][j]

			# now that we have the R matrix, we can compute the numerical sum for that candidate and put it in the output file
			f2.write(str(sum) + "\n")
			
			# now we just have to write it into the file
			full = 0  # reset full for the next candidate

# original unlabelled image names
with open(settings.search_name) as csvfile:  # 'search_name_caltech-256.csv'
	readCSV = csv.reader(csvfile, delimiter=',')
	n = 0
	for row in readCSV:
		# this will ignore the original instance of the image (we only need the patch's probabilities to create the vector)
		if n == 0:
			f1.write(str(row[0]) + "\n")
			n = n + 1
		else:
			if n == 15:
				n = 0
			else:
				n = n + 1
		
f1.close()
f2.close()
print("done")
