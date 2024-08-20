import shutil
parent = '/home/reddy16/datafolder_AEN'
parent2 = '/home/reddy16/models'
parent3 = '/home/reddy16/BSHG_EB4'
parent4 = '/home/reddy16/BSHG_EB4_valid'
parent5 = '/home/reddy16/BSHG_EB4_test'
parent6 = '/home/reddy16/FSHG_EB4'
parent7 = '/home/reddy16/FSHG_EB4_valid'
parent8 = '/home/reddy16/FSHG_EB4_test'
parent9 = '/home/reddy16/TPEF_EB4'
parent10 = '/home/reddy16/TPEF_EB4_valid'
parent11 = '/home/reddy16/TPEF_EB4_test'
shutil.rmtree(parent)
shutil.rmtree(parent2)
shutil.rmtree(parent3)
shutil.rmtree(parent4)
shutil.rmtree(parent5)
shutil.rmtree(parent6)
shutil.rmtree(parent7)
shutil.rmtree(parent8)
shutil.rmtree(parent9)
shutil.rmtree(parent10)
shutil.rmtree(parent11)

import os

os.mkdir('/home/reddy16/BSHG_EB4')
os.mkdir('/home/reddy16/BSHG_EB4/FA')
os.mkdir('/home/reddy16/BSHG_EB4/FTC')
os.mkdir('/home/reddy16/BSHG_EB4_test')
os.mkdir('/home/reddy16/BSHG_EB4_test/FA')
os.mkdir('/home/reddy16/BSHG_EB4_test/FTC')
os.mkdir('/home/reddy16/BSHG_EB4_valid')
os.mkdir('/home/reddy16/BSHG_EB4_valid/FA')
os.mkdir('/home/reddy16/BSHG_EB4_valid/FTC')

os.mkdir('/home/reddy16/FSHG_EB4')
os.mkdir('/home/reddy16/FSHG_EB4/FA')
os.mkdir('/home/reddy16/FSHG_EB4/FTC')
os.mkdir('/home/reddy16/FSHG_EB4_test')
os.mkdir('/home/reddy16/FSHG_EB4_test/FA')
os.mkdir('/home/reddy16/FSHG_EB4_test/FTC')
os.mkdir('/home/reddy16/FSHG_EB4_valid')
os.mkdir('/home/reddy16/FSHG_EB4_valid/FA')
os.mkdir('/home/reddy16/FSHG_EB4_valid/FTC')

os.mkdir('/home/reddy16/TPEF_EB4')
os.mkdir('/home/reddy16/TPEF_EB4/FA')
os.mkdir('/home/reddy16/TPEF_EB4/FTC')
os.mkdir('/home/reddy16/TPEF_EB4_test')
os.mkdir('/home/reddy16/TPEF_EB4_test/FA')
os.mkdir('/home/reddy16/TPEF_EB4_test/FTC')
os.mkdir('/home/reddy16/TPEF_EB4_valid')
os.mkdir('/home/reddy16/TPEF_EB4_valid/FA')
os.mkdir('/home/reddy16/TPEF_EB4_valid/FTC')

os.mkdir('/home/reddy16/models')
os.mkdir('/home/reddy16/datafolder_AEN')
os.mkdir('/home/reddy16/datafolder_AEN/resultant_images')
os.mkdir('/home/reddy16/datafolder_AEN/BSHG_images')
os.mkdir('/home/reddy16/datafolder_AEN/FSHG_images')
os.mkdir('/home/reddy16/datafolder_AEN/TPEF_images')

os.mkdir('/home/reddy16/datafolder_AEN/resultant_images/trainData')
os.mkdir('/home/reddy16/datafolder_AEN/BSHG_images/trainData')
os.mkdir('/home/reddy16/datafolder_AEN/FSHG_images/trainData')
os.mkdir('/home/reddy16/datafolder_AEN/TPEF_images/trainData')

os.mkdir('/home/reddy16/datafolder_AEN/resultant_images/valData')
os.mkdir('/home/reddy16/datafolder_AEN/BSHG_images/valData')
os.mkdir('/home/reddy16/datafolder_AEN/FSHG_images/valData')
os.mkdir('/home/reddy16/datafolder_AEN/TPEF_images/valData')

os.mkdir('/home/reddy16/datafolder_AEN/resultant_images/testData')
os.mkdir('/home/reddy16/datafolder_AEN/BSHG_images/testData')
os.mkdir('/home/reddy16/datafolder_AEN/FSHG_images/testData')
os.mkdir('/home/reddy16/datafolder_AEN/TPEF_images/testData')

os.mkdir('/home/reddy16/datafolder_AEN/resultant_images/trainData/FA')
os.mkdir('/home/reddy16/datafolder_AEN/BSHG_images/trainData/FA')
os.mkdir('/home/reddy16/datafolder_AEN/FSHG_images/trainData/FA')
os.mkdir('/home/reddy16/datafolder_AEN/TPEF_images/trainData/FA')

os.mkdir('/home/reddy16/datafolder_AEN/resultant_images/valData/FA')
os.mkdir('/home/reddy16/datafolder_AEN/BSHG_images/valData/FA')
os.mkdir('/home/reddy16/datafolder_AEN/FSHG_images/valData/FA')
os.mkdir('/home/reddy16/datafolder_AEN/TPEF_images/valData/FA')

os.mkdir('/home/reddy16/datafolder_AEN/resultant_images/testData/FA')
os.mkdir('/home/reddy16/datafolder_AEN/BSHG_images/testData/FA')
os.mkdir('/home/reddy16/datafolder_AEN/FSHG_images/testData/FA')
os.mkdir('/home/reddy16/datafolder_AEN/TPEF_images/testData/FA')

os.mkdir('/home/reddy16/datafolder_AEN/resultant_images/trainData/FTC')
os.mkdir('/home/reddy16/datafolder_AEN/BSHG_images/trainData/FTC')
os.mkdir('/home/reddy16/datafolder_AEN/FSHG_images/trainData/FTC')
os.mkdir('/home/reddy16/datafolder_AEN/TPEF_images/trainData/FTC')

os.mkdir('/home/reddy16/datafolder_AEN/resultant_images/valData/FTC')
os.mkdir('/home/reddy16/datafolder_AEN/BSHG_images/valData/FTC')
os.mkdir('/home/reddy16/datafolder_AEN/FSHG_images/valData/FTC')
os.mkdir('/home/reddy16/datafolder_AEN/TPEF_images/valData/FTC')

os.mkdir('/home/reddy16/datafolder_AEN/resultant_images/testData/FTC')
os.mkdir('/home/reddy16/datafolder_AEN/BSHG_images/testData/FTC')
os.mkdir('/home/reddy16/datafolder_AEN/FSHG_images/testData/FTC')
os.mkdir('/home/reddy16/datafolder_AEN/TPEF_images/testData/FTC')




