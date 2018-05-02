import os
import sys
from subprocess import call

class ArgumentMissingException(Exception):
    def __init__(self):
        print("usage: {} <dirname>".format(sys.argv[0]))
        sys.exit(1)
    
def check_path(path):
    	return bool(os.path.exists(path)) #Checkif path exists

def main(path):

    if check_path(path):

        for f in os.listdir(path): #Return list of files in path directory
            file = os.path.splitext(os.path.basename(f))[0]
            split = ''.join([i for i in file if not i.isdigit()])
            head, sep, tail = split.partition('.')
            print (head)
            writer = open(path + '/' + f,'a')
            writer.write(head)
            writer.close()

            #file = open(f,'a')

if __name__ == '__main__': #Execute all code before reading source file, ie. execute import, evaluate def to equal name to main
    if len(sys.argv) != 2: # Count number of arguments which contains the command-line arguments passed to the script if it is not equal to 2 ie for (py main.py 1_arg 2_arg)
        raise ArgumentMissingException
    path = sys.argv[1] #python main.py "path_to/img_dir" ie the argv[1] value
    path = os.path.abspath(path) #Accesing filesystem for Return a normalized absolutized version of the pathname path
    main(path)
    #s = memento(path)
    #s.main(path) # Def main to path
