# Error Cases (Auto Extracted)

## 1. numeric mismatch
- question: what percentage of total facilities as measured in square feet are leased?
- gold: 0.14
- pred: 73.6
- raw_output: [FINAL_ANSWER]73.6[/FINAL_ANSWER]

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
- pred: 9.7
- raw_output: [FINAL_ANSWER]9.7[/FINAL_ANSWER]

## 6. percent scaling
- question: if 2014 underlying operating profit increases at the same pace as 2013 , what would it be , in millions?
- gold: 2186.0
- pred: 5.73
- raw_output: To calculate the 2014 underlying operating profit if it increases at the same pace as 2013, we first need to determine the rate of increase from 2012 to 2013.

The increase from 2012 to 2013 is:
$2098 - $2014 = $84 million

The rate of increase is:
$84 million / $2014 = 0.0417 (or 4.17%)

To calculate the 2014 underlying operating profit if it increases at the same pace as 2013, we apply the same rate of increase to the 2013 underlying operating profit:
$2098 * 0.0417 = $87.73 million

Adding this to the 2013 underlying operating profit gives:
$2098 + $87.73 = $2185.73 million

[FINAL_ANSWER]2185.73[/FINAL_ANSWER]

## 7. percent scaling
- question: what percentage of long-term debt is current debt?
- gold: 0.077
- pred: 0.0767
- raw_output: To find the percentage of long-term debt that is current debt, we need to calculate the ratio of current debt to total long-term debt and then multiply by 100 to get the percentage.

Current debt = Long-term debt of payments due by period less than 1 year = 510
Total long-term debt = 6652

Percentage of current debt = (Current debt / Total long-term debt) * 100
Percentage of current debt = (510 / 6652) * 100 ≈ 7.67%

[FINAL_ANSWER]7.67[/FINAL_ANSWER]

## 8. percent scaling
- question: what are the pre-tax earnings in 2016 , in billions?
- gold: 1.1
- pred: 1.136
- raw_output: To calculate the pre-tax earnings in 2016, we can use the information that pre-tax earnings in 2017 were $1.42 billion, which is 25% higher than in 2016.

Let $ x $ be the pre-tax earnings in 2016. Then, the pre-tax earnings in 2017 can be expressed as:

$$
x + 0.25x = 1.42
$$

$$
1.25x = 1.42
$$

$$
x = \frac{1.42}{1.25}
$$

$$
x = 1.136
$$

Therefore, the pre-tax earnings in 2016 were approximately $1.136 billion.

[FINAL_ANSWER]1.136[/FINAL_ANSWER]

## 9. percent scaling
- question: what is the percent change in expected volatility between 2012 and 2013?
- gold: -0.087
- pred: -7.92
- raw_output: To calculate the percent change in expected volatility between 2012 and 2013, we use the formula:

$$
\text{Percent Change} = \left( \frac{\text{New Value} - \text{Old Value}}{\text{Old Value}} \right) \times 100
$$

Where:
- New Value = Expected volatility in 2013 = 22.1%
- Old Value = Expected volatility in 2012 = 24.0%

$$
\text{Percent Change} = \left( \frac{22.1 - 24.0}{24.0} \right) \times 100 = \left( \frac{-1.9}{24.0} \right) \times 100 \approx -7.92\%
$$

[FINAL_ANSWER]-7.92[/FINAL_ANSWER]

## 10. percent scaling
- question: based on the december 31 2018 target what was the debt to equity ratio
- gold: 4.6
- pred: 0.82
- raw_output: [FINAL_ANSWER]82%[/FINAL_ANSWER]
