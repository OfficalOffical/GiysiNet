
import editCsv
from model import mainModel

width = 224
height = 224
nRowSetter = 20000 #modelde de manuel gir

def getImageFromCSV(csvPath,mode):


    csv, imarr = editCsv.getAndCleanCsv(csvPath, width, height, nRowSetter)

    mainModel(csv['categoryx_id'],imarr,mode)


    return csv





