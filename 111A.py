# the package which used to read the excel
import xlrd
import numpy
import string
import plotly
import re
plotly.tools.set_credentials_file(username='moyan.melody', api_key='4lV27cTNrjqYQj14ch7P')
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score

data = xlrd.open_workbook('games-features.xlsx') # open the xls file
table = data.sheets()[0]                     # open the first sheet
nrows = 601
ncolumns = table.ncols
a = u'True'
matrix = numpy.zeros((500, 9))
genre = 0                                   # in the first column of the matrix
platform = 0
age = 0
test = numpy.zeros((100,9))
evaluate = 0
vector = numpy.zeros(500)
puref = numpy.zeros((500, 6))
puret = numpy.zeros((100, 6))
tin = 0
pop = numpy.zeros(500)
for i in range(nrows):
    if i == 0:
        continue
    for j in range(19, 30):

        if table.row_values(i)[j] == a:
            genre += 1
        if i >= 501:
            test[tin][0] = genre
            puret[tin][0] = genre
        else:

            matrix[i-1][0] = genre             # put into the first column
            puref[i-1][0] = genre

        # print table.row_values(i-1)[j]
        
    for k in range(12, 15):
        if table.row_values(i)[k] == a:
            platform += 1
        if i >= 501:
            test[tin][1] = platform
            puret[tin][1] = platform
        else:
            matrix[i-1][1] = platform          # put into the second column
            puref[i-1][1] = platform
        # print(table.row_values(0)[k])

    age = table.row_values(i)[3]
    if age == 0:
        if i >= 501:
            test[tin][2] = 3
            puret[tin][2] = 3
        else:
            matrix[i-1][2] = 3                 # ages put into the third column
            puref[i-1][2] = 3
    elif age < 17 :
        if i >= 501:
            test[tin][2] = 2
            puret[tin][2] = 2
        else:
            matrix[i-1][2] = 2
            puref[i - 1][2] = 2
    elif age >= 17:
        if i >= 501:
            test[tin][2] = 1
            puret[tin][2] = 1
        else:
            matrix[i-1][2] = 1
            puref[i - 1][2] = 1

    language = table.row_values(i)[31]     #only consider the top eight languages
    regex = re.compile(r'[%s\s]+' % re.escape(string.punctuation))
    sp = regex.split(language)
    length = sp.__len__()
    length = length-1
    mlang = 0
    while length >= 0:
         if sp[length] == u'English':
             mlang += 1
         if sp[length] == u'French':
             mlang += 1
         if sp[length] == u'Chinese':
             mlang += 1
         if sp[length] == u'Spanish':
             mlang += 1
         if sp[length] == u'Arabic':
             mlang += 1
         if sp[length] == u'Russian':
             mlang += 1
         if sp[length] == u'German':
             mlang += 1
         if sp[length] == u'Japanese':
             mlang += 1
         length -= 1
    length = 0
    if i >= 501:
        test[tin][3] = mlang
        puret[tin][3] = mlang
    else:
        matrix[i-1][3] = mlang                       # put into the fourth column
        puref[i - 1][3] = mlang

    # print(sp)
    # print(mlang)
    mlang = 0

    # if table.row_values(i)[10] == u'true':       # whether free     ######## ???
    #    matrix[i-1][4] = 1                       # put into the fourth column

    price = table.row_values(i)[30]
    if price == 0:                               # prices put into the fifth column
        if i >= 501:
            test[tin][4] = 5
            puret[tin][4] = 5
        else:
            matrix[i-1][4] = 5
            puref[i - 1][4] = 5
    elif price<= 10 :
        if i >= 501:
            test[tin][4] = 4
            puret[tin][4] = 4
        else:
            matrix[i-1][4] = 4
            puref[i - 1][4] = 4
    elif price <= 20:
        if i >= 501:
            test[tin][4] = 3
            puret[tin][4] = 3
        else:
            matrix[i-1][4] = 3
            puref[i - 1][4] = 3
    elif price <= 30:
        if i >= 501:
            test[tin][4] = 2
            puret[tin][4] = 2
        else:
            matrix[i-1][4] = 2
            puref[i - 1][4] = 2
    elif price > 30:
        if i >= 501:
            test[tin][4] = 1
            puret[tin][4] = 1
        else:
            matrix[i-1][4] = 1
            puref[i-1][4] = 1
        # print(matrix[i-1][1])
    #print(table.row_values(i)[16])
    if table.row_values(i)[16] == a:      # whether can be played by multiplayer
        print(table.row_values(i)[16])
        if i >= 501:
            test[tin][5] = 1
            puret[tin][5] = 1
        else:
            matrix[i-1][5] = 1
            puref[i-1][5] = 1
            #print(table.row_values(i)[16])

    # the popularity calculated by recommendation/players
    if i >= 501:
        test[tin][6] = table.row_values(i)[4]/table.row_values(i)[7]
    else:
        matrix[i-1][6] = table.row_values(i)[4]/table.row_values(i)[7]
        pop[i-1] = matrix[i-1][6]

    if i>= 501:
        test[tin][7] = table.row_values(i)[4]
    else:
        matrix[i-1][7] = table.row_values(i)[4]
        #print(table.row_values(i)[4])
    # print(matrix[i-1][6])
    ######## need to remove all the recommendation is 0

    genre = 0
    platform = 0
    if i >= 501:
        tin += 1

diff = numpy.zeros(100)
axis = numpy.zeros(100)
evaluate = 0
for j in range(100):
    axis[j] = j
    for i in range(500):

        distance = 0
        for a in range(6): # genre platform age language price multiplayers
            distance += (matrix[i][a]-test[j][a]) ** 2
            evaluate = distance ** (0.5)
        if evaluate == 0:
           evaluate = 999
        vector[i] = evaluate
        #print(evaluate)
        #print(i)

    final = numpy.zeros(3)

    if(final[0] == 0):
        final[0] = vector[0]
    #print(final[0])
    #print(vector[0])

    if(final[1] == 0):
        final[1] = vector[0]

    if(final[2] == 0):
        final[2] = vector[0]

    best = numpy.zeros(3)
    for i in range(500):

        if(vector[i]<final[0]):
            #print(vector[i])
            final[2] = vector[i]
            final[1] = vector[i]
            final[0] = vector[i]
            best[0] = i
        if(vector[i]<final[1]):
            final[2] = final[1]
            final[1] = vector[i]
            best[1] = i
        if(vector[i]<final[2]):
            final[2] = vector[i]
            best[2] = i

    diff[j] = test[j][6] - matrix[best[0]][6]
    #print(diff[j])


##### create the graph
N = 100
#print(axis)
#print(diff)
large = numpy.zeros(500)
rate = numpy.zeros(500)
#for i in range(500):
#   large[i] = i
#    rate[i] = matrix[i][6]

#print(large)
#print(rate)
#trace0 = go.Scatter(
#    x = large,
#    y = rate,
#    mode = 'markers',
#    name = 'markers'
#)
#data = [trace0]
#py.plot(data, filename='line-mode')


############# specify the class
for i in range(500):
    #print(matrix[i][6])
    if matrix[i][6] < 0.007:

        matrix[i][8] = 1
    elif matrix[i][6] < 0.012:
        matrix[i][8] = 2
    else:
        matrix[i][8] = 3

for i in range(100):
    if test[i][6] < 0.007:
        test[i][8] = 1
    elif test[i][6] < 0.012:
        test[i][8] = 2
    else:
        test[i][8] = 3

cla = numpy.zeros(500)
for i in range(500):
    cla[i] = matrix[i][8]

# using the svm
clf = svm.SVC()
#print(puref)
#print(cla)
clf.fit(puref, cla)

c = clf.predict([puret[0]])
svmdiff = numpy.zeros(100)
svmind = numpy.zeros(100)
for i in range(100):
    svmdiff[i] = 999

for j in range(100):

    for i in range(500):
        distance = 0
        for a in range(6): # genre platform age language price multiplayers
            if matrix[i][8] == c[0]:
                distance += (matrix[i][a]-test[j][a]) ** 2
                evaluate = distance ** (0.5)
        if evaluate == 0:
            evaluate = 999
        if svmdiff[j] > evaluate:
           svmdiff[j] = evaluate
           svmind[j] = i
        #print(i)

#print(svmind)
svmf = numpy.zeros(100)
for i in range(100):
    axis[i] = i
    svmf[i] = test[i][6] - matrix[svmind[i]][6]




#print(svmf.mean())

#trace0 = go.Scatter(
#    x = axis,
#    y = svmf,
#    mode = 'lines',
#    name = 'lines'
#)
#data = [trace0]
#py.plot(data, filename='line-mode')

########### calculate the weights of each of the features
# six features overall
#eff = numpy.zeros(6)
eff = np.linalg.lstsq(puref, pop)[0]
print(eff)
effdiff = numpy.zeros(100)
for i in range(100):
    for j in range(6):
        predict = eff[j]*test[i][j]
    effdiff[i] = test[i][6] - predict

for i in range(100):
    svmf[i] = abs(svmf[i])
    diff[i] = abs(diff[i])
#print(svmf.std())
#print(diff.std())
#trace0 = go.Scatter(
#    x = axis,
#    y = effdiff,
#    mode = 'lines+markers',
#    name = 'lines+markers'
#)
#data = [trace0]
#py.plot(data, filename='effdiff')