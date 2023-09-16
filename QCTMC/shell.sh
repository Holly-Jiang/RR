#DEGREE: 1, HEIGHT: 1-10 11-100 101-500 501-1000,  CONFLICT: 1/ISOLATE: 0, SINGLE SIGNAL: 0
python3 BatchObservingExpression.py 1 1 10  1 0 > results/degree1height10.txt
python3 BatchObservingExpression.py 1 1 10 0  0 >> results/degree1height10.txt
python3 BatchObservingExpression.py 1 11 100   1  0 >  results/degree1height100.txt
python3 BatchObservingExpression.py 1 11 100   0  0 >>   results/degree1height100.txt
python3 BatchObservingExpression.py 1 101 500 1  0 >   results/degree1height500.txt
python3 BatchObservingExpression.py 1 101 500  0  0 >>  results/degree1height500.txt
python3 BatchObservingExpression.py 1 501 1000 1  0 >  results/degree1height1000.txt
python3 BatchObservingExpression.py 1 501 1000  0  0 >>  results/degree1height1000.txt
#DEGREE: 2, HEIGHT: 1-10 11-100 101-500 501-1000, CONFLICT: 1/ISOLATE: 0, SINGLE SIGNAL: 0
python3 BatchObservingExpression.py 2 1 10  1  0 >   results/degree2height10.txt
python3 BatchObservingExpression.py 2 1 10 0 0  >>   results/degree2height10.txt
python3 BatchObservingExpression.py 2 11 100 1  0 >   results/degree2height100.txt
python3 BatchObservingExpression.py 2 11 100  0  0 >>  results/degree2height100.txt
python3 BatchObservingExpression.py 2 101 500  1  0 >   results/degree2height500.txt
python3 BatchObservingExpression.py 2 101 500  0  0 >> results/degree2height500.txt
python3 BatchObservingExpression.py 2 501 1000  1  0 >  results/degree2height1000.txt
python3 BatchObservingExpression.py 2 501 1000 0  0 >>  results/degree2height1000.txt
#DEGREE: 3, HEIGHT: 1-10 11-100 101-500 501-1000, CONFLICT: 1/ISOLATE: 0, SINGLE SIGNAL: 0
python3 BatchObservingExpression.py 3 1 10  1  0 >   results/degree3height10.txt
python3 BatchObservingExpression.py 3 1 10 0  0 >>   results/degree3height10.txt
python3 BatchObservingExpression.py 3 11 100  1  0 >   results/degree3height100.txt
python3 BatchObservingExpression.py 3 11 100  0  0 >>  results/degree3height100.txt
python3 BatchObservingExpression.py 3 101 500  1  0 >   results/degree3height500.txt
python3 BatchObservingExpression.py 3 101 500  0  0 >> results/degree3height500.txt
python3 BatchObservingExpression.py 3 501 1000  1  0 >  results/degree3height1000.txt
python3 BatchObservingExpression.py 3 501 1000 0  0 >>  results/degree3height1000.txt
#DEGREE: 4, HEIGHT: 1-10 11-100 101-500 501-1000, CONFLICT: 1/ISOLATE: 0, SINGLE SIGNAL: 0
python3 BatchObservingExpression.py 4 1 10  1  0 >   results/degree4height10.txt
python3 BatchObservingExpression.py 4 1 10 0  0 >>   results/degree4height10.txt
python3 BatchObservingExpression.py 4 11 100  1  0 >   results/degree4height100.txt
python3 BatchObservingExpression.py 4 11 100  0  0 >> results/degree4height100.txt
python3 BatchObservingExpression.py 4 101 500  1  0 >   results/degree4height500.txt
python3 BatchObservingExpression.py 4 101 500  0  0 >> results/degree4height500.txt
python3 BatchObservingExpression.py 4 501 1000  1  0 >  results/degree4height1000.txt
python3 BatchObservingExpression.py 4 501 1000 0  0 >>  results/degree4height1000.txt
#DEGREE: 1, HEIGHT: 1-10 11-100 101-500 501-1000, CONFLICT: 1/ISOLATE: 0, CNF OF MULTIPLE SIGNALS: 1
python3 BatchObservingExpression.py 1 1 10 1 1 > results/booldegree1height10.txt
python3 BatchObservingExpression.py 1 1 10 0  1 >> results/booldegree1height10.txt
python3 BatchObservingExpression.py 1 11 100 1 1 >  results/booldegree1height100.txt
python3 BatchObservingExpression.py 1 11 100 0  1 >>  results/booldegree1height100.txt
python3 BatchObservingExpression.py 1 101 500 1  1 >  results/booldegree1height500.txt
python3 BatchObservingExpression.py 1 101 500 0  1 >>  results/booldegree1height500.txt
python3 BatchObservingExpression.py 1 501 1000 1 1 >  results/booldegree1height1000.txt
python3 BatchObservingExpression.py 1 501 1000 0  1 >>  results/booldegree1height1000.txt
#DEGREE: 2, HEIGHT: 1-10 11-100 101-500 501-1000, CONFLICT: 1/ISOLATE: 0, CNF OF MULTIPLE SIGNALS: 1
python3 BatchObservingExpression.py 2 1 10 1  1 >  results/booldegree2height10.txt
python3 BatchObservingExpression.py 2 1 10 0  1 >>  results/booldegree2height10.txt
python3 BatchObservingExpression.py 2 11 100 1  1 >  results/booldegree2height100.txt
python3 BatchObservingExpression.py 2 11 100 0  1 >>  results/booldegree2height100.txt
python3 BatchObservingExpression.py 2 101 500 1  1 >  results/booldegree2height500.txt
python3 BatchObservingExpression.py 2 101 500 0  1 >>  results/booldegree2height500.txt
python3 BatchObservingExpression.py 2 501 1000 1   1 >  results/booldegree2height1000.txt
python3 BatchObservingExpression.py 2 501 1000 0  1 >>  results/booldegree2height1000.txt
#DEGREE: 3, HEIGHT: 1-10 11-100 101-500 501-1000, CONFLICT: 1/ISOLATE: 0, CNF OF MULTIPLE SIGNALS: 1
python3 BatchObservingExpression.py 3 1 10  1  1 >   results/booldegree3height10.txt
python3 BatchObservingExpression.py 3 1 10 0  1 >>   results/booldegree3height10.txt
python3 BatchObservingExpression.py 3 11 100  1  1 >   results/booldegree3height100.txt
python3 BatchObservingExpression.py 3 11 100  0  1 >>  results/booldegree3height100.txt
python3 BatchObservingExpression.py 3 101 500  1  1 >   results/booldegree3height500.txt
python3 BatchObservingExpression.py 3 101 500  0  1 >> results/booldegree3height500.txt
python3 BatchObservingExpression.py 3 501 1000  1  1 >  results/booldegree3height1000.txt
python3 BatchObservingExpression.py 3 501 1000 0  1 >>  results/booldegree3height1000.txt
#DEGREE: 4, HEIGHT: 1-10 11-100 101-500 501-1000, CONFLICT: 1/ISOLATE: 0, CNF OF MULTIPLE SIGNALS: 1
python3 BatchObservingExpression.py 4 1 10  1  1 >   results/booldegree4height10.txt
python3 BatchObservingExpression.py 4 1 10 0  1 >>   results/booldegree4height10.txt
python3 BatchObservingExpression.py 4 11 100  1  1 >   results/booldegree4height100.txt
python3 BatchObservingExpression.py 4 11 100  0  1 >>  results/booldegree4height100.txt
python3 BatchObservingExpression.py 4 101 500  1  1 >   results/booldegree4height500.txt
python3 BatchObservingExpression.py 4 101 500  0  1  >> results/booldegree4height500.txt
python3 BatchObservingExpression.py 4 501 1000  1  1 >  results/booldegree4height1000.txt
python3 BatchObservingExpression.py 4 501 1000 0  1  >>  results/booldegree4height1000.txt
