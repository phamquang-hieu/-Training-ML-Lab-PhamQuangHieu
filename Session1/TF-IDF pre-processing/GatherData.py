from os import listdir
from os.path import isfile
import re
import sys

def gather20NewsGroupsData():
	path = "datasets/20news-bydate/"
	dirs = [path + dirName + '/' for dirName in listdir(path) if not isfile(path+dirName)]

	# prevent wrong in order of the two datasets in the root directory
	trainDir, testDir = (dirs[0], dirs[1]) if 'train' in dirs[0] else (dirs[1], dirs[0])

	# List of all groups 
	listNewsGroups = [newsGroup for newsGroup in listdir(trainDir)]
	listNewsGroups.sort()

	with open('datasets/20news-bydate/stop-words.txt') as f:
		stop_words = f.read().splitlines()

	from nltk.stem.porter import PorterStemmer

	stemmer = PorterStemmer()

	def collectDataFrom(parentDir, newsGroupList):
		"""collect data from all files in a directory"""
		data = []
		for groupId, newsGroup in enumerate(newsGroupList):
			label = groupId
			dirPath = parentDir + '/' + newsGroup + '/'
			files = [(fileName, dirPath + fileName) 
						for fileName in listdir(dirPath)
						if isfile(dirPath+fileName)] 
			files.sort()

			for fileName, filePath in files:
				with open(filePath) as f:
					text = f.read().lower()
					print(text)
					exit()
					words = [stemmer.stem(word) 
							for word in re.split(r"\W+", text)
							if word not in stop_words]
					content = ' '.join(words)
					# splitlines() function will split a string into lists.
					# the spliting is done at line breaks
					# guarantee that there is no extra line break after spliting
					assert len(content.splitlines()) == 1 
					data.append(str(label) + '<fff>' + fileName + '<fff>' + content)
		return data

	trainData = collectDataFrom(
		parentDir=trainDir, 
		newsGroupList = listNewsGroups
	)

	testData = collectDataFrom(
		parentDir=testDir,
		newsGroupList= listNewsGroups
	)
	fullData = trainData + testData

	with open("datasets/20news-bydate/20news-train-processed.txt", "w") as f:
		f.write('\n'.join(trainData))

	with open("datasets/20news-bydate/20news-test-processed.txt", "w") as f:
		f.write('\n'.join(testData))

	with open("datasets/20news-bydate/20news-full-processed.txt", "w") as f:
		f.write('\n'.join(fullData))

# gather20NewsGroupsData()
		
			




# gather20NewsGroupsData()