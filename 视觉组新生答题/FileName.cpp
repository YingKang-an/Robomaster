/* TEST 1-1 */

#include<iostream>
using namespace std;

int main()
{
    int i,j;
    for (i = 1; i <= 9; ++i)
    {
        for (j = 1; j <= i; ++j)
        {
            cout << j << "*" << i << "=" << i * j << " ";
        }
        cout << endl;
    }

    return 0;
}


/* TEST 1-2 */

#include<iostream>
using namespace std;

int main()
{
    int i = 1;
    while (i <= 9)
    {
        int j = 1;
        while (j <= i)
        {
            cout << j << "*" << i << "=" << j * i << " ";
            ++j;
        }
        cout << endl;
        ++i;
    }
    return 0;
}

/* TEST 2 */

#include<iostream>
using namespace std;

int num;

void fun(int array[], int length)
{
    int min_number = 0;

    for (int j = 1; j < length; ++j)
    {
        if (array[j] < array[min_number])
        {
            min_number = j;
        }
    }
    
    int end_result = array[min_number];
    array[min_number] = array[length - 1];
    array[length - 1] = end_result;

}

int main()
{
    const int length = 100;
    int array[length];
    cout << "Enter the quantity of numbers£º"<<endl;
    cin >> num;
    
    if (num <= 0 || num > length)
    {
        cout << "Iiiegal input!" << endl;
        return -1;
    }
    
    cout << "Enter " << num << " numbers" << endl;
    for (int i = 0; i < num; ++i)
    {
        cin >> array[i];
    }

    fun(array, num);
 
    cout << "The array after swapping the minimum value and the last bit is: " << endl;
    for (int m = 0; m < num; ++m)
    {
        cout << array[m] << " ";

    }
 
    return 0;
}


/* OPTIONAL TEST 4 */

#include <iostream>
using namespace std;

int main()
{
    int electric;
    cout << "Enter the electricity consumption for this month:";
    cin >> electric;

    double money = 0;

    if (electric <= 150)
        money = electric * 0.44630;
    else if (electric <= 400)
        money = 150 * 0.44630 + (electric - 150) * 0.46630;
    else
        money = 150 * 0.44630 + 250 * 0.46630 + (electric - 400) * 0.56630;

    money = int(money * 10 + 0.5) / 10.0;

    cout << money << " yuan"<<endl;

    return 0;
}
