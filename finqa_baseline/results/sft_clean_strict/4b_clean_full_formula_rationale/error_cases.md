# Error Cases (Auto Extracted)

## 1. numeric mismatch
- question: what percentage of total facilities as measured in square feet are leased?
- gold: 0.14
- pred: 73.61
- raw_output: [FINAL_ANSWER]73.61[/FINAL_ANSWER]

## 2. numeric mismatch
- question: what is the percentage change in cash flow hedges in 2011 compare to the 2010?
- gold: 0.099
- pred: 10.0
- raw_output: [FINAL_ANSWER]10.0[/FINAL_ANSWER]

## 3. numeric mismatch
- question: what portion of total purchase price is related to stock awards?
- gold: 0.028999999999999998
- pred: 121.4
- raw_output: [FINAL_ANSWER]121.4[/FINAL_ANSWER]

## 4. numeric mismatch
- question: what was the difference in percentage cumulative total shareholder return on masco common stock versus the s&p 500 index for the five year period ended 2017?
- gold: None
- pred: 123.45
- raw_output: [FINAL_ANSWER]123.45[/FINAL_ANSWER]

## 5. numeric mismatch
- question: what was the percentage change in total rental expense under operating leases from july 2 , 2005 to july 1 , 2006?
- gold: 0.07
- pred: 9.67
- raw_output: [FINAL_ANSWER]9.67[/FINAL_ANSWER]

## 6. percent scaling
- question: what was the percentage change in the impact of the euro on earnings from 2010 to 2011?
- gold: 0.42
- pred: 0.4194
- raw_output: To calculate the percentage change in the impact of the euro on earnings from 2010 to 2011, we use the formula:

Percentage change = ((New value - Old value) / Old value) * 100

From the context, the impact of the euro on earnings in 2010 is 18.6, and in 2011 it is 26.4.

Percentage change = ((26.4 - 18.6) / 18.6) * 100
Percentage change = (7.8 / 18.6) * 100
Percentage change = 0.4193548387 * 100
Percentage change = 41.93548387%

Rounded to two decimal places, the percentage change is 41.94%.

[FINAL_ANSWER]41.94[/FINAL_ANSWER]

## 7. percent scaling
- question: what was the percentage change in receivables from brokers with in other current assets from 2009 to 2010?
- gold: 0.29
- pred: 27.27
- raw_output: To calculate the percentage change in receivables from brokers in other current assets from 2009 to 2010, we use the formula:

Percentage change = ((New value - Old value) / Old value) * 100

Where:
- New value (2010) = 11.2
- Old value (2009) = 8.8

Percentage change = ((11.2 - 8.8) / 8.8) * 100
Percentage change = (2.4 / 8.8) * 100
Percentage change = 0.2727 * 100
Percentage change = 27.27%

[FINAL_ANSWER]27.27[/FINAL_ANSWER]

## 8. percent scaling
- question: if 2014 underlying operating profit increases at the same pace as 2013 , what would it be , in millions?
- gold: 2186.0
- pred: 3.0
- raw_output: To calculate the 2014 underlying operating profit if it increases at the same pace as 2013, we first need to determine the rate of increase from 2012 to 2013. 

The increase from 2012 to 2013 is $2098 - $2014 = $84 million.

The rate of increase is $84 million / $2014 = 0.0417 or 4.17%.

Applying this rate to the 2013 underlying operating profit, the 2014 underlying operating profit would be $2098 * (1 + 0.0417) = $2183 million.

[FINAL_ANSWER]2183[/FINAL_ANSWER]

## 9. percent scaling
- question: what percentage of long-term debt is current debt?
- gold: 0.077
- pred: 0.0767
- raw_output: To find the percentage of long-term debt that is current debt, we need to calculate the ratio of current debt to total long-term debt and then multiply by 100.

Current debt = Payments due by period less than 1 year = 510
Total long-term debt = 6652

Percentage of current debt = (Current debt / Total long-term debt) * 100
Percentage of current debt = (510 / 6652) * 100
Percentage of current debt ≈ 7.67%

[FINAL_ANSWER]7.67[/FINAL_ANSWER]

## 10. percent scaling
- question: what are the pre-tax earnings in 2016 , in billions?
- gold: 1.1
- pred: 1.136
- raw_output: To calculate the pre-tax earnings in 2016, we can use the information that pre-tax earnings in 2017 were $1.42 billion, which is 25% higher than in 2016. 

Let $ x $ be the pre-tax earnings in 2016. Then, the pre-tax earnings in 2017 can be expressed as $ x + 0.25x = 1.25x $. We know that $ 1.25x = 1.42 $ billion.

Solving for $ x $, we get:
$$
x = \frac{1.42}{1.25} = 1.136
$$

Therefore, the pre-tax earnings in 2016 were approximately $1.136 billion.

[FINAL_ANSWER]1.136[/FINAL_ANSWER]
