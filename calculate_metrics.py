'''
big thanks to:
https://github.com/bigbigbean/Segmentation-Evaluation-by-SimpleITK/blob/master/quality.py (helpful tips for sitk)
https://www.sciencedirect.com/science/article/pii/S0031320321000443#sec0001 (efficient algorithm for calculating 3d hausdorff)
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4533825/ (general formulas and summaries of different evaluation metrics)
'''
import time
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import os
import pandas as pd
import multiprocessing
import time
from typing import Dict, List

# subject dictionary in the form of key: subject name, value: [ground_truth_image_filepath, prediction_image_filepath]
SUBLIST = dict()
SUBLIST['203011'] = ['test segmentation files/203011_gt_11label.nii', 'test segmentation files/203011_GRACE_11label.nii']

class Comparator:
    # constructor
    def __init__(self):
        self.images = dict() # dict of dicts
        self.results = dict()
        self.globalNumTissues: int = 0

    # turns files into nib objects
    def readImagesFromFiles(self, fileDict: Dict[str, List[str]]) -> bool:
        print('Reading images from files...')
        for subName, imageList in fileDict.items():
            # checking if file exists
            if not os.path.exists(imageList[0]):
                print(f'{imageList[0]} does not exist.')
                print('Aborting...')
                return False
            if not os.path.exists(imageList[1]):
                print(f'{imageList[1]} does not exist.')
                print('Aborting...')
                return False
            
            # generating nib ground truth and prediction images
            gtImage = nib.load(imageList[0])
            gtImage = gtImage.get_fdata()
            gtImage = gtImage.astype(np.uint8)

            predImage = nib.load(imageList[1])
            predImage = predImage.get_fdata()
            predImage = predImage.astype(np.uint8)

            # saving resulting images
            self.images[subName] = dict()
            self.images[subName]['gtImage'] = gtImage
            self.images[subName]['predImage'] = predImage

            # preparing results for later
            self.results[subName] = dict()

        return True

    # turns files into nib objects, FIXME: finish this if needed
    def readImagesFromFolderPattern(self, folderPath, delimiter='-'):
        pass

    # calculates the number of tissues each subject has, assuming they all have the same amount
    def calcNumTissues(self) -> None:
        # assumes that all the subjects have the same number of tissue types
        maxNum, minNum = np.max(self.images['203011']['gtImage']), np.min(self.images['203011']['gtImage'])
        self.globalNumTissues = maxNum - minNum + 1

    def _createStringNum(self, num: int) -> str:
        maxLength = len(str(self.globalNumTissues))
        strNum = str(num)
        while len(strNum) != maxLength:
            strNum = '0' + strNum
        
        return strNum

    # turns nib objects into dice coefficients
    def computeDice(self) -> None:
        print('Calculating dice...')
        start = time.time()
        diceComputer = sitk.LabelOverlapMeasuresImageFilter()
        for subName, subDict in self.images.items():
            # turns nib image object into sitk image object
            labelPred = sitk.GetImageFromArray(subDict['predImage'], isVector=False)
            labelTrue = sitk.GetImageFromArray(subDict['gtImage'], isVector=False)

            # does calculations
            diceComputer.Execute(labelPred, labelTrue)

            # records results
            self.results[subName]['total dice'] = diceComputer.GetDiceCoefficient()
            for i in range(self.globalNumTissues):
                self.results[subName]['dice ' + self._createStringNum(i)] = diceComputer.GetDiceCoefficient(i)
        end = time.time()
        print(f'Elapsed time: {end - start}s')

    # runs convolution on image to get borders of mask
    def convolve(self, image) -> np.ndarray:
        # kernel to detect edge voxels
        kernel = [[[0, 0, 0],
                   [0, 1, 0],
                   [0, 0, 0]],
                   
                   [[0, 1, 0],
                    [1, 2, 1],
                    [0, 1, 0]],
                    
                    [[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]]]
        # loading in images and changing to correct data type
        sitkImage = sitk.GetImageFromArray(image)
        sitkImage = sitk.Cast(sitkImage, sitk.sitkFloat64)
        sitkKernel = sitk.GetImageFromArray(kernel)
        sitkKernel = sitk.Cast(sitkKernel, sitk.sitkFloat64)

        convolutionCalculator = sitk.ConvolutionImageFilter()

        # convolving image with convolution
        convolved_image = convolutionCalculator.Execute(sitkImage, sitkKernel)
        convolved_image = sitk.GetArrayFromImage(convolved_image)

        # maintaining that 0's in original image will be 0 in final image
        convolved_image[image==0] = 0

        # setting values back to 0's and 1's
        condition = (convolved_image == 8) | (convolved_image == 0)
        convolved_image = np.where(condition, 0, 1)
        return convolved_image
    
    # updates terminal to show program is running during hausdorff calculation, currently not used
    def _hausdorffTerminalUpdater(self) -> None:
        i: int = 1
        while not self.hausdorffFinished:
            os.system('cls')
            ellip = '.' * i
            print('Calculating Hausdorff Distance' + ellip)
            i = i + 1 if not i == 3 else 1
            time.sleep(0.5)

    # actual function that does early break hausdorff
    def _ebDirectedHausdorff(self, A, B) -> float:
        try:
            # want the max of all the min distances
            cmax = 0
            # sum of all minimums, for avg hausdorff calculation
            minSum = 0
            # take out the union of the two images, where they both equal 1
            E = A - B
            E = np.where(E < 0, 0, E)
            # leaves only the edges of the masks, don't know if this is necessary?
            E = self.convolve(E)
            B = self.convolve(B)
            bx, by, bz = np.where(B==1)
            ex, ey, ez = np.where(E==1)
            bIndices = list(zip(bx, by, bz))
            bIndices = np.array(bIndices)
            eIndices = list(zip(ex, ey, ez))
            iteration = 0
            for eIndex in eIndices:
                differences = np.array(eIndex) - bIndices
                distances = np.linalg.norm(differences, axis=1)
                minDistance = np.min(distances)
                cmax = minDistance if (minDistance > cmax) else cmax
                minSum += minDistance
                iteration += 1
            # finds num voxels, assuming both images have the same shape
            voxels = np.count_nonzero(A == 1)
            print(voxels) # FIXME: remove this when finalizing
            minSum /= voxels
            return cmax, minSum
        except:
            print('error')
            return -1, -1
    
    # helper function to calc hausdorff distance
    def _getHausdorff(self, subName, i, image1, image2, outerQueue) -> None:
        h1, avgh1 = self._ebDirectedHausdorff(image1, image2)
        h2, avgh2 = self._ebDirectedHausdorff(image2, image1)
        maxHausdorff = max(h1, h2)
        avgHausdorff = (avgh1 + avgh2) / 2
        result = (subName, f'hausdorff {self._createStringNum(i)}', maxHausdorff)
        outerQueue.put(result)
        result = (subName, f'average hausdorff {self._createStringNum(i)}', avgHausdorff)
        outerQueue.put(result)    
    
    # computes hausdorff using self-created method
    def selfComputeHausdorff(self) -> None:
        print('Computing hausdorff distance...')
        start = time.time()
        processList = []
        outerQueue = multiprocessing.Queue() # each item is (subName, metricName, metricValue)
        for subName, subDict in self.images.items():
            for i in range(self.globalNumTissues):
                binMaskgt = np.zeros(subDict['gtImage'].shape)
                binMaskpred = np.zeros(subDict['predImage'].shape)

                binMaskgt[subDict['gtImage']==i] = 1
                binMaskpred[subDict['predImage']==i] = 1

                process = multiprocessing.Process(target=self._getHausdorff, args=(subName, i, binMaskgt, binMaskpred, outerQueue))
                process.start()
                processList.append(process)
        
        for process in processList:
            process.join()

        while not outerQueue.empty():
            item = outerQueue.get_nowait()
            if item is None:
                break
            self.results[item[0]][item[1]] = item[2]
        end = time.time()
        print(f'Elapsed time: {end - start}s')

    # just produce hausdorff of one mask of one subject, used primarily for debugging and testing
    def selfOneHausdorffMask(self) -> None:
        binMaskgt = np.zeros(self.images['203011']['gtImage'].shape)
        binMaskpred = np.zeros(self.images['203011']['predImage'].shape)

        binMaskgt[self.images['203011']['gtImage']==1] = 1
        binMaskpred[self.images['203011']['predImage']==1] = 1

        outerQueue = multiprocessing.Queue()

        self._getHausdorff('203011', 1, binMaskgt, binMaskpred, outerQueue)

        item = outerQueue.get()
        self.results[item[0]][item[1]] = item[2]

        item = outerQueue.get()
        self.results[item[0]][item[1]] = item[2]

    # calculates jaccard coefficient with self algorithm, currently not used due to greater computational time
    def _calcJaccardCoefficient(self) -> None:
        print('Calculating jaccard coefficient...')
        start = time.time()
        for subName, subDict in self.images.items():
            gtImage = subDict['gtImage']
            predImage = subDict['predImage']

            for i in range(self.globalNumTissues):
                Ax, Ay, Az = np.where(gtImage == i)
                Bx, By, Bz = np.where(predImage == i)

                ACoords = set(zip(Ax, Ay, Az))
                BCoords = set(zip(Bx, By, Bz))

                magIntersection = len(ACoords & BCoords)
                magUnion = len(ACoords | BCoords)

                self.results[subName]['self jaccard coefficient ' + self._createStringNum(i)] = magIntersection / magUnion

        end = time.time()
        print(f'Elapsed time: {end - start}s')

    # calculates jaccard coefficient with sitk, more efficient
    def calcJaccardCoefficient(self) -> None:
        print('Calculating jaccard coefficient...')
        start = time.time()
        for subName, subDict in self.images.items():
            gtImage = subDict['gtImage']
            predImage = subDict['predImage']

            for i in range(self.globalNumTissues):
                binMaskgt = np.where(gtImage == i, 1, 0)
                binMaskpred = np.where(predImage == i, 1, 0)

                binMaskgt = sitk.GetImageFromArray(binMaskgt)
                binMaskpred = sitk.GetImageFromArray(binMaskpred)

                jaccardComputer = sitk.LabelOverlapMeasuresImageFilter()

                jaccardComputer.Execute(binMaskgt, binMaskpred)

                self.results[subName]['jaccard coefficient ' + self._createStringNum(i)] = jaccardComputer.GetJaccardCoefficient()

        end = time.time()
        print(f'Elapsed time: {end - start}s')

    # calculates volumetric similarity, currently not used
    def calcVolumetricSimilarity(self) -> None:
        print('Calculating volumetric similarity...')
        start = time.time()
        for subName, subDict in self.images.items():
            gtImage = subDict['gtImage']
            predImage = subDict['predImage']

            for i in range(self.globalNumTissues):
                # uses formula found in 3rd link at the top of this file
                magA = np.count_nonzero(gtImage == i)
                magB = np.count_nonzero(predImage == i)

                vs = 1 - ((magA - magB) / (magA + magB))

                self.results[subName]['volumetric similarity ' + self._createStringNum(i)] = abs(vs)
        end = time.time()
        print(f'Elapsed time: {end - start}s')
    
    # prints data with a panda dataframe, not currently used, doesn't display well in terminal, depricated
    def printPandaResults(self) -> None:
        data = dict()
        data['Subject Name'] = []

        # populates lists so that corresponding indices are the same subject
        for subName, subDict in self.results.items():
            data['Subject Name'].append(subName)
            for metricName, metricValue in subDict.items():
                if metricName in data:
                    data[metricName].append(metricValue)
                else:
                    data[metricName] = []
                    data[metricName].append(metricValue)

        # creates DataFrame object
        df = pd.DataFrame(data)

        # displays results more nicely
        pd.set_option('display.colheader_justify', 'center')
        pd.set_option('display.expand_frame_repr', False)
        print(df.to_string(index=False))

    # prints calculated metrics to terminal
    def printTerminal(self) -> None:
        dashLength = 40
        print()
        print('Results:')
        for subName, subDict in self.results.items():
            print(u'\u2500' * dashLength) # prints horizontal divider
            print(f'Subject Name: {subName}')
            sortedKeys = sorted(subDict.keys())
            for metricName in sortedKeys:
                print(f'{metricName}: {subDict[metricName]}')
        print(u'\u2500' * dashLength) # prints horizontal divider

if __name__ == '__main__':
    print('Running...')
    start = time.time()
    myComparator = Comparator()

    # reading images
    if not myComparator.readImagesFromFiles(SUBLIST):
        quit()

    myComparator.calcNumTissues()

    # generating metrics
    myComparator.computeDice()
    myComparator.calcJaccardCoefficient()
    myComparator.selfComputeHausdorff()
    # myComparator.selfOneHausdorffMask()

    # displaying metrics in a nicer(?) manner
    myComparator.printTerminal()

    print('Finished.')
    end = time.time()
    print(f'Total elapsed time: {end - start}s')