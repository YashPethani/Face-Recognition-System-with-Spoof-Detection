import numpy as np
import face_recognition

picture_of_me = face_recognition.load_image_file("ronak.jpeg")
print(face_recognition.face_encodings(picture_of_me))
my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

np.save('ronak_face.npy', my_face_encoding)

ronak = np.array([-1.04919702e-01, 4.51847017e-02, -5.05765378e-02, -3.83374467e-02,
                  -2.72736400e-02, -2.01149844e-02, -1.61499083e-02, -1.20000608e-01,
                  2.08538264e-01, -8.93117636e-02, 1.61548361e-01, -1.99284144e-02,
                  -1.82469726e-01, -5.65734133e-02, -1.51991844e-01, 4.28384021e-02,
                  -1.66980013e-01, -1.38422921e-01, -4.04210836e-02, -1.57744601e-01,
                  -1.35256117e-03, 3.21823657e-02, 3.04202680e-02, 5.07865883e-02,
                  -1.57994419e-01, -3.04825902e-01, -9.74115506e-02, -2.08582088e-01,
                  8.58064815e-02, -8.47528279e-02, 1.56357363e-02, 2.06128526e-02,
                  -1.61234096e-01, 7.74325058e-03, 1.28589068e-02, 1.06676877e-01,
                  -6.92079263e-03, 7.56631233e-03, 2.33196199e-01, 3.93134318e-02,
                  -8.22516754e-02, -7.65669718e-03, 3.24816853e-02, 3.53298903e-01,
                  1.10586852e-01, 1.01819590e-01, -1.24050342e-02, -5.16120344e-02,
                  1.26177043e-01, -2.36015156e-01, 1.42683715e-01, 1.31801233e-01,
                  1.15920246e-01, 4.05537412e-02, 9.24973115e-02, -1.94475561e-01,
                  5.73989935e-03, 4.05321755e-02, -1.82971433e-01, 9.37277600e-02,
                  3.73111758e-03, -4.26955894e-03, -5.67865670e-02, 1.40838604e-02,
                  2.69202083e-01, 1.22701764e-01, -1.39241844e-01, -2.71717533e-02,
                  1.62473619e-01, -2.36952454e-01, 4.10364792e-02, 6.10601455e-02,
                  -4.31717560e-02, -1.62331060e-01, -2.08785966e-01, 8.90795439e-02,
                  5.37295282e-01, 1.70815468e-01, -1.02355883e-01, 2.98539773e-02,
                  -9.48313102e-02, -5.99683858e-02, 1.09958075e-01, 5.69739826e-02,
                  -1.74593583e-01, 3.23062316e-02, -9.33352113e-02, 4.53415178e-02,
                  1.59353971e-01, 4.97628003e-04, -8.29316229e-02, 1.45061359e-01,
                  -2.26403978e-02, 6.76396713e-02, 1.74541529e-02, 1.14599206e-02,
                  -1.36046499e-01, -1.21505754e-02, -4.52841148e-02, 7.47919362e-03,
                  6.13869429e-02, -6.93896413e-02, -5.45657240e-03, 5.44008911e-02,
                  -1.68737233e-01, 1.15876928e-01, 2.68548671e-02, -1.18266627e-01,
                  -7.94334337e-03, 7.02663511e-02, -1.05740607e-01, -6.40174225e-02,
                  1.68161824e-01, -3.24357003e-01, 1.88108250e-01, 1.46066830e-01,
                  2.69625001e-02, 2.51383543e-01, 4.51094955e-02, 4.57431227e-02,
                  2.41612811e-02, -1.63366217e-02, -1.13300852e-01, -9.67966765e-02,
                  -4.27811500e-03, -2.14770213e-02, 6.50588572e-02, 1.25133861e-02])


mayur = np.array([-0.11441386, 0.05793351, 0.0657698, -0.02465041, 0.03785457,
                  -0.02075242, -0.0383823, -0.06185715, 0.14292161, -0.07372424,
                  0.19134916, -0.04599359, -0.1841362, -0.13891995, 0.01412673,
                  0.04787168, -0.05145019, -0.17451471, -0.05810031, -0.14860123,
                  -0.06394535, 0.01649462, 0.05452134, 0.00086475, -0.21563415,
                  -0.37790731, -0.07678348, -0.17263518, -0.00865261, -0.04857092,
                  -0.00040412, 0.07091781, -0.1950492, -0.03635065, 0.02563301,
                  0.0965694, 0.04252435, 0.0625818, 0.20077442, -0.00715665,
                  -0.06855965, 0.00180559, 0.07257449, 0.29960737, 0.12546752,
                  0.05448982, 0.0175046, 0.00221466, 0.09412148, -0.17689075,
                  0.12628528, 0.09021811, 0.11141208, 0.0275098, 0.16164057,
                  -0.10786736, 0.00807155, 0.00605601, -0.18406282, 0.09202126,
                  -0.02316232, 0.02152376, -0.09008974, -0.01097175, 0.2738744,
                  0.09941136, -0.07137731, -0.11354341, 0.18761335, -0.13866612,
                  0.03875915, 0.08889976, -0.0837847, -0.10361049, -0.23832195,
                  0.10724206, 0.38733858, 0.11375485, -0.15219903, 0.11828248,
                  -0.08172259, -0.07581056, 0.01709411, -0.03959195, -0.16450374,
                  0.09169964, -0.05245414, 0.07075649, 0.18499969, 0.06930766,
                  -0.0805407, 0.14671978, -0.04400258, 0.05786302, 0.09424118,
                  -0.03897231, -0.10077408, -0.04013304, -0.1486142, -0.02160572,
                  0.09392967, -0.09107014, 0.02290076, 0.07083558, -0.19806688,
                  0.09945752, -0.00681603, -0.04985024, -0.00114545, 0.17653397,
                  -0.1313554, -0.06487489, 0.13772838, -0.30975896, 0.15464778,
                  0.08743317, 0.03671006, 0.13833518, 0.07043333, 0.02139484,
                  0.01551396, 0.0585371, -0.09632352, -0.09337648, 0.03179381,
                  -0.02149671, 0.09036399, 0.10041111])

pooja = np.array([-2.03361750e-01, 1.02429494e-01, -1.69355869e-02, -9.33491513e-02,
                  -3.76082957e-02, -1.55632254e-02, 1.21758189e-02, -7.35830665e-02,
                  2.22886711e-01, -5.79887703e-02, 2.14842588e-01, -4.86370809e-02,
                  -1.68292567e-01, -6.03597462e-02, -5.59216924e-02, 1.61012977e-01,
                  -1.98429108e-01, -1.61396191e-01, -2.48198509e-02, -4.96902838e-02,
                  1.30213946e-01, -7.41129294e-02, 2.04473792e-04, 1.36821344e-01,
                  -1.94680646e-01, -3.96182925e-01, -8.93443897e-02, -1.02590546e-01,
                  3.33014429e-02, -8.79479125e-02, 5.47905080e-02, 1.40198156e-01,
                  -2.45291188e-01, -5.75092621e-03, -2.81139743e-02, 1.20560586e-01,
                  2.65875012e-02, -1.98434833e-02, 1.79849729e-01, -5.94770256e-03,
                  -2.56009340e-01, -5.22226766e-02, 9.83302966e-02, 2.80281633e-01,
                  1.89382970e-01, 4.87636104e-02, 2.07695477e-02, -4.43579406e-02,
                  6.41092807e-02, -2.03495547e-01, 6.54484779e-02, 1.74280882e-01,
                  8.80464315e-02, 4.34840247e-02, 3.77399884e-02, -1.08308643e-01,
                  -6.76588062e-03, 6.83336556e-02, -2.16774553e-01, -1.94647536e-02,
                  -2.73561478e-03, -7.69761503e-02, -5.32742962e-02, -9.16405618e-02,
                  1.99517936e-01, 1.37552619e-01, -1.26457989e-01, -7.07852095e-02,
                  2.34700337e-01, -1.33791983e-01, 7.46361492e-03, 7.42056444e-02,
                  -1.23589419e-01, -2.31849939e-01, -2.93464661e-01, 3.60318460e-02,
                  4.24880177e-01, 1.69328213e-01, -1.02775551e-01, 1.12527952e-01,
                  -8.94255266e-02, -5.49794212e-02, 1.55546367e-02, 1.57265306e-01,
                  -6.97913319e-02, 6.43677637e-02, -4.69138920e-02, 1.23609938e-01,
                  1.37746841e-01, 2.46578325e-02, -1.18620964e-02, 2.08119586e-01,
                  -2.60657500e-02, -1.99359730e-02, -2.75557078e-02, 9.92363244e-02,
                  -1.12627476e-01, 6.28975639e-03, -1.28493816e-01, -8.28748420e-02,
                  3.14622670e-02, 4.41198088e-02, 1.50088640e-03, 1.21041402e-01,
                  -1.97045341e-01, 1.18554734e-01, 4.03214358e-02, -2.15386748e-02,
                  -2.24230867e-02, 5.34514450e-02, -1.92102939e-02, -8.53729770e-02,
                  5.97549304e-02, -2.38935143e-01, 1.11344703e-01, 1.54022723e-01,
                  -4.35981564e-02, 1.75320879e-01, 7.00903758e-02, 8.74220654e-02,
                  4.23764735e-02, -1.05004795e-01, -1.75671205e-01, -5.59940189e-03,
                  1.03383757e-01, -1.51416538e-02, 6.78806603e-02, 9.90182813e-03])