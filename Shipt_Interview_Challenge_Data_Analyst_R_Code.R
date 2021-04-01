# Importing the libraries
library(dplyr)
library(tidyr)
library(lubridate)
library(ggplot2)
library(stringr)


# Reading the InterviewData_Cost.csv
InterviewData_Cost <- read.csv('/Users/abhilashasinha/Downloads/interview_challenge_data_analyst/InterviewData_Cost.csv')

# Data in InterviewData_Cost
head(InterviewData_Cost)
 
# Reading the InterviewData_Rev.csv file
InterviewData_Rev <- read.csv('/Users/abhilashasinha/Downloads/interview_challenge_data_analyst/InterviewData_Rev.csv')

# Data in InterviewData_Rev
head(InterviewData_Rev)

# Q1. Join these two data sets by “date” and “source_id”, returning all rows from both regardless of whether there is a match between the two data sets
# Join the two tables using outer join
data_outer <- merge(x = InterviewData_Cost, y = InterviewData_Rev, by = c("date","source_id"), all = TRUE)

# Result of outer join
head(data_outer)

# Q2. Join these two data sets by “date” and “source_id”, returning only the rows from the “Cost” file that have no corresponding date in the “Revenue” file.
# Join the two tables using left join
data_left <- merge(x = InterviewData_Cost, y = InterviewData_Rev, by = c("date","source_id"), all.x = TRUE)

# Result of left join
head(data_left)

# Q3. Using your result from #1: a. What are the Top 4 sources (“source_id” values) in terms of total revenue generation across this data set?
# Replacing NA with 0
data_outer_replace <- na.replace(data_outer, 0)

# Aggrerating the revenue for each source_id
data_outer_total <- data_outer_replace %>% group_by(source_id) %>% summarize(summary_variable=sum(revenue))

# Result of aggregation
head(data_outer_total)

# Sorting the output of aggregation in descending order
data_outer_sort <- data_outer_total[order(-data_outer_total$summary_variable),]

# Selecting the top 4 Source_ids
data_outer_top4 <- head(data_outer_sort$source_id,4)

# Top 4 Source_ids
data_outer_top4

# b. How would you visualize the monthly revenue for those Top 4 sources?
# Selecting records which belong to the top 4 sources with highest revenue
data_source4 <- data_outer %>% filter(source_id %in% data_outer_top4)

# Replacing NA with 0
data_source4_re <- na.replace(data_source4, 0)

# Data with records belonging to the top 4 source_ids
head(data_source4_re)

# Checking the data type of columns
str(data_source4_re)

# Converting column 'date' to date format
data_source_type <- data_source4_re
data_source_type$date <-as.factor(data_source_type$date)
data_source_type$date <-strptime(data_source_type$date,format="%m/%d/%Y")
data_source_type$date<-as.Date(data_source_type$date,format="%m/%d/%Y")

# Data after data type conversion
head(data_source_type)

# Selection only 3 columns (date, source_id, revenue) and dropping column 'cost'
data_source_rev <- data_source_type[,-3]

# Grouping the columns according to month and source_id
data_grouped_col <- data_source_rev %>% group_by("Month"=month(date), source_id) %>% summarize(summary_revenue=sum(revenue))

# Data in grouped column
head(data_grouped_col)

# Plot of monthly revenue
revenue_plot <- ggplot(data = data_grouped_col, aes(x = Month, y = summary_revenue, color = source_id)) + geom_line()
revenue_plot

# Q4.Assuming you’ve read the data into an R object called activity_data, run the following code to build a basic logistic regression model:
# Reading the InterviewData_Activity.csv
activity_data <- read.csv('/Users/abhilashasinha/Downloads/interview_challenge_data_analyst/InterviewData_Activity.csv')

# Data in activity_data
head(activity_data)

# Running logistic regression code
full_logit_model <- glm(formula = active ~ age + gender + metropolitan_area + device_type, data = activity_data,family = binomial(link = "logit"))

# Applying the model on activity_data to assess the prediction accuracy
glm.probs <- predict(full_logit_model, newdata = activity_data, type = "response")

# Converting probability to binary category
glm.pred <- ifelse(glm.probs > 0.5, "1", "0")

# Confusion matrix
table(glm.pred,active)

# Accuracy of the model
mean(glm.pred == activity_data$active)

# The accuracy of the model is 58%


# Q5. Split the data into training and test samples, and build a model over the training data using the following R code:
# Splitting the data into training and test sets
training_data <- activity_data[1:4000,]
test_data <- activity_data[4001:5420,]

# Running logistic regression code
training_logit_model <- glm(formula = active ~ age + gender + metropolitan_area + device_type, data = training_data, family = binomial(link = "logit"))

# Applying the model on test_data to assess the prediction accuracy
glm_test.probs <- predict(training_logit_model, newdata = test_data, type = "response")

# Converting probability to binary category
glm_test.pred <- ifelse(glm_test.probs > 0.5, "1", "0")

# Confusion matrix
table(glm_test.pred, test_data$active)

# Accuracy of the model
mean(glm_test.pred == test_data$active)


# (6) This data comes from a subset of userdata JSON blobs stored in our database. Parse out the values (stored in the “data_to_parse” column) into four separate columns. So for example, the four additional columns for the first entry would have values of “N”, “U”, “A7”, and “W”.

# Reading the InterviewData_Parsing.csv file
InterviewData_Parsing <- read.csv('/Users/abhilashasinha/Downloads/interview_challenge_data_analyst/InterviewData_Parsing.csv')

# Data in InterviewData_Parsing
head(InterviewData_Parsing)

# Removing the special characters
InterviewData_Parsing$data_to_parse <- str_replace_all(InterviewData_Parsing$data_to_parse, "[[:punct:]]", " ")

# Data after removing the special characters
head(InterviewData_Parsing)

# Dividing the data in separate columns
InterviewData_Parsing <- InterviewData_Parsing %>% separate(data_to_parse, c("Sep", "Value", "Col1","Col2", "Col3", "Col4"))

# Data in the InterviewData_Parsing
head(InterviewData_Parsing)
