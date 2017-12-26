import socket

sourceroot = '/home/mlpa/data_ssd/workspace/github/AutoencodingTheWorld'
datasetroot = '/home/mlpa/data_ssd/workspace/dataset/CVC-ClinicDB'

# set paths according to the host computer
hostname = socket.gethostname()
if 'mlpa-1070x2' == hostname:
    sourceroot = '/home/mlpa/data_ssd/workspace/github/AutoencodingTheWorld'
    datasetroot = '/home/mlpa/data_ssd/workspace/dataset/CVC-ClinicDB'
elif 'leejeyeol-System-Product-Name':
    sourceroot = '/home/leejeyeol/git/AutoencodingTheWorld'
    datasetroot = '/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB'


#()()
#('')HAANJU.YOO
