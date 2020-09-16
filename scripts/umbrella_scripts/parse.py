# Reads the replica matix without the time in the first coloumn.And, tracks the position for all 40 windows.  
import math
import sys
from tempfile import TemporaryFile

class parse: # 'Holds all information for a given list'
    def __init__(self, filename):
        self.filename = filename
		 
    def textparse(self):# 'Main module to parse the list'
        Net_list = open(self.filename, 'r')
        line_net = [line.strip('\n\r\t') for line in Net_list]  # 'Splits invidual line'
        Net_list.close()
	row = []
	column = []

        for k in range(0,40):
		if (k == 0):
			f0 = open('output%d.txt' %(k),'w')
		elif (k == 1):
			f1 = open('output%d.txt' %(k),'w')
		elif (k == 2):
			f2 = open('output%d.txt' %(k),'w')
		elif (k == 3):
			f3 = open('output%d.txt' %(k),'w')
		elif (k == 4):
			f4 = open('output%d.txt' %(k),'w')
		elif (k == 5):
			f5 = open('output%d.txt' %(k),'w')	
		elif (k == 6):
			f6 = open('output%d.txt' %(k),'w')
		elif (k == 7):
			f7 = open('output%d.txt' %(k),'w')
		elif (k == 8):
			f8 = open('output%d.txt' %(k),'w')
		elif (k == 9):
			f9 = open('output%d.txt' %(k),'w')
		elif (k == 10):
			f10 = open('output%d.txt' %(k),'w')
		elif (k == 11):
			f11 = open('output%d.txt' %(k),'w')
		elif (k == 12):
			f12 = open('output%d.txt' %(k),'w')
		elif (k == 13):
			f13 = open('output%d.txt' %(k),'w')
		elif (k == 14):
			f14 = open('output%d.txt' %(k),'w')
		elif (k == 15):
			f15 = open('output%d.txt' %(k),'w')	
		elif (k == 16):
			f16 = open('output%d.txt' %(k),'w')
		elif (k == 17):
			f17 = open('output%d.txt' %(k),'w')
		elif (k == 18):
			f18 = open('output%d.txt' %(k),'w')
		elif (k == 19):
			f19 = open('output%d.txt' %(k),'w')
		elif (k == 20):
			f20 = open('output%d.txt' %(k),'w')
		elif (k == 21):
			f21 = open('output%d.txt' %(k),'w')
		elif (k == 22):
			f22 = open('output%d.txt' %(k),'w')
		elif (k == 23):
			f23 = open('output%d.txt' %(k),'w')
		elif (k == 24):
			f24 = open('output%d.txt' %(k),'w')
		elif (k == 25):
			f25 = open('output%d.txt' %(k),'w')	
		elif (k == 26):
			f26 = open('output%d.txt' %(k),'w')
		elif (k == 27):
			f27 = open('output%d.txt' %(k),'w')
		elif (k == 28):
			f28 = open('output%d.txt' %(k),'w')
		elif (k == 29):
			f29 = open('output%d.txt' %(k),'w')
		elif (k == 30):
			f30 = open('output%d.txt' %(k),'w')
		elif (k == 31):
			f31 = open('output%d.txt' %(k),'w')
		elif (k == 32):
			f32 = open('output%d.txt' %(k),'w')
		elif (k == 33):
			f33 = open('output%d.txt' %(k),'w')
		elif (k == 34):
			f34 = open('output%d.txt' %(k),'w')
		elif (k == 35):
			f35 = open('output%d.txt' %(k),'w')	
		elif (k == 36):
			f36 = open('output%d.txt' %(k),'w')
		elif (k == 37):
			f37 = open('output%d.txt' %(k),'w')
		elif (k == 38):
			f38 = open('output%d.txt' %(k),'w')
		elif (k == 39):
			f39 = open('output%d.txt' %(k),'w')
		
		for i in range(0, len(line_net) - 1 ):
            		line_new_net = [' '.join(line_net[i].split())]  # 'Removes empty spaces'
            		line_elem = [elem.strip().split(' ') for elem in line_new_net]  

            		line_elem_2 = list(line_elem[0])    # 'Makes individual entry as a list'
            
            		for j in xrange(0, len(line_elem_2)):  # 'Checks for a particular elements location'

				if (line_elem_2[j] ==str(k) ):
					if (line_elem_2[j] == '0'):
						f0.write(str(i)+ ' ')
                                                f0.write(str(j) +'\n')
                                                #f0.write('('+ str(i)+ ',')
						#f0.write(str(j)+ ')\n')
					elif (line_elem_2[j] == '1'):
						f1.write(str(i)+ ' ')
						f1.write(str(j)+ '\n')
					elif (line_elem_2[j] == '2'):
						f2.write(str(i)+ ' ')
						f2.write(str(j)+ '\n')
					elif (line_elem_2[j] == '3'):
						f3.write( str(i)+ ' ')
						f3.write(str(j)+ '\n')
					elif (line_elem_2[j] == '4'):
						f4.write( str(i)+ ' ')
						f4.write(str(j)+ '\n')
					elif (line_elem_2[j] == '5'):
						f5.write(str(i)+ ' ')
						f5.write(str(j)+ '\n')
					elif (line_elem_2[j] == '6'):
						f6.write(str(i)+ ' ')
						f6.write(str(j)+ '\n')
					elif (line_elem_2[j] == '7'):
						f7.write(str(i)+ ' ')
						f7.write(str(j)+ '\n')
					elif (line_elem_2[j] == '8'):
						f8.write(str(i)+ ' ')
						f8.write(str(j)+ '\n')
					elif (line_elem_2[j] == '9'):
						f9.write(str(i)+ ' ')
						f9.write(str(j)+ '\n')
					elif (line_elem_2[j] == '10'):
						f10.write(str(i)+ ' ')
						f10.write(str(j)+ '\n')
					elif (line_elem_2[j] == '11'):
						f11.write(str(i)+ ' ')
						f11.write(str(j)+ '\n')
					elif (line_elem_2[j] == '12'):
						f12.write( str(i)+ ' ')
						f12.write(str(j)+ '\n')
					elif (line_elem_2[j] == '13'):
						f13.write( str(i)+ ' ')
						f13.write(str(j)+ '\n')
					elif (line_elem_2[j] == '14'):
						f14.write(str(i)+ ' ')
						f14.write(str(j)+ '\n')
					elif (line_elem_2[j] == '15'):
						f15.write(str(i)+ ' ')
						f15.write(str(j)+ '\n')
					elif (line_elem_2[j] == '16'):
						f16.write(str(i)+ ' ')
						f16.write(str(j)+ '\n')
					elif (line_elem_2[j] == '17'):
						f17.write(str(i)+ ' ')
						f17.write(str(j)+ '\n')
					elif (line_elem_2[j] == '18'):
						f18.write(str(i)+ ' ')
						f18.write(str(j)+ '\n')
					elif (line_elem_2[j] == '19'):
						f19.write(str(i)+ ' ')
						f19.write(str(j)+ '\n')
					elif (line_elem_2[j] == '20'):
						f20.write(str(i)+ ' ')
						f20.write(str(j)+ '\n')
					elif (line_elem_2[j] == '21'):
						f21.write( str(i)+ ' ')
						f21.write(str(j)+ '\n')
					elif (line_elem_2[j] == '22'):
						f22.write(str(i)+ ' ')
						f22.write(str(j)+ '\n')
					elif (line_elem_2[j] == '23'):
						f23.write(str(i)+ ' ')
						f23.write(str(j)+ '\n')
					elif (line_elem_2[j] == '24'):
						f24.write(str(i)+ ' ')
						f24.write(str(j)+ '\n')
					elif (line_elem_2[j] == '25'):
						f25.write(str(i)+ ' ')
						f25.write(str(j)+ '\n')
					elif (line_elem_2[j] == '26'):
						f26.write(str(i)+ ' ')
						f26.write(str(j)+ '\n')
					elif (line_elem_2[j] == '27'):
						f27.write(str(i)+ ' ')
						f27.write(str(j)+ '\n')
					elif (line_elem_2[j] == '28'):
						f28.write(str(i)+ ' ')
						f28.write(str(j)+ '\n')
					elif (line_elem_2[j] == '29'):
						f29.write(str(i)+ ' ')
						f29.write(str(j)+ '\n')
					elif (line_elem_2[j] == '30'):
						f30.write(str(i)+ ' ')
						f30.write(str(j)+ '\n')
					elif (line_elem_2[j] == '31'):
						f31.write(str(i)+ ' ')
						f31.write(str(j)+ '\n')
					elif (line_elem_2[j] == '32'):
						f32.write(str(i)+ ' ')
						f32.write(str(j)+ '\n')
					elif (line_elem_2[j] == '33'):
						f33.write(str(i)+ ' ')
						f33.write(str(j)+ '\n')
					elif (line_elem_2[j] == '34'):
						f34.write(str(i)+ ' ')
						f34.write(str(j)+ '\n')
					elif (line_elem_2[j] == '35'):
						f35.write(str(i)+ ' ')
						f35.write(str(j)+ '\n')
					elif (line_elem_2[j] == '36'):
						f36.write( str(i)+ ' ')
						f36.write(str(j)+ '\n')
					elif (line_elem_2[j] == '37'):
						f37.write(str(i)+ ' ')
						f37.write(str(j)+ '\n')
					elif (line_elem_2[j] == '38'):
						f38.write(str(i)+ ' ')
						f38.write(str(j)+ '\n')
					elif (line_elem_2[j] == '39'):
						f39.write(str(i)+ ' ')
						f39.write(str(j)+ '\n')

       
