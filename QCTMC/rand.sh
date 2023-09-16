for i in {1..4}
do
python3 random_test.py $i 1 10 0 1 20  >> random_test_1_10_0.txt
python3 random_test.py $i 11 100 0 1 200 >> random_test_11_100_0.txt
python3 random_test.py $i 101 500 0 11 1000 >> random_test_101_500_0.txt
python3 random_test.py $i 501 1000 0 100 1000 >> random_test_501_1000_0.txt
done
for i in {1..4}
do
python3 random_test.py $i 1 10 1 1 20 >> bool_random_test_1_10_1.txt
python3 random_test.py $i 11 100 1  1 200 >> bool_random_test_11_100_1.txt
python3 random_test.py $i 101 500 1  11 1000 >> boo_random_test_101_500_1.txt
python3 random_test.py $i 501 1000 1  100 1000 >> bool_random_test_501_1000_1.txt
done



python3 random_test.py 3 501 1000 1  100 1000 >> bool_random_test_501_1000_1.txt