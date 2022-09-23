% Lab 1 - Task 2: Build a naive Bayes classifier
% Lab 1 - Task 3: Improve the classifier with Laplace (additive) smoothing

function [classification,errorRate]=myBayesClassifier(trainingSet,dataSet,a)


%% check for errors
if( size(dataSet,2) < (size(trainingSet,2)-1))
    error("Invalid inputs: the number of columns of dataSet must be at least the number of the column of the trainingSet - 1");
end

if(sum(any(dataSet<1))>0 || sum(any(trainingSet<1))>0)
    error("Invalid inputs: some elements are <1");
end

% variables for Laplace smoothing
numberOfPossibleValues = [3,3,2,2];


%% training

%init variables
numVar = size(trainingSet,2)-1;
classes = unique(trainingSet(:,end));
numClasses = size(classes,1);
P=cell(numClasses,numVar);
levelsName=cell(numClasses,numVar);

for i=1:numVar
    for j=1:numClasses
        P{j,i}=zeros(1,size(unique(trainingSet(:,i)),1));
    end
end

%compute the conditioned probabilities
for i = 1:numClasses
    occurrences = sum(trainingSet(:,end)==classes(i));
    for j = 1:numVar
        levels = unique(trainingSet(:,j));
        for k = 1:size(levels)
            P{i,j}(1,k) = (sum(trainingSet(:,j)==levels(k) & trainingSet(:,numVar+1)==classes(i))+a)/(occurrences+(a*numberOfPossibleValues(j)));
            levelsName{i,j}(1,k)=levels(k);
        end
    end
end

%% Classification

numRows=size(dataSet,1);
probabilities=ones(1,numClasses);
classification=zeros(numRows,1);

%compute discriminant function
for i=1:numRows
    for j=1:numClasses
        probability=sum(trainingSet(:,end)==classes(j))/size(trainingSet,1);
        for k=1:numVar
            probability=probability*P{j,k}(1,dataSet(i,k));
        end
        probabilities(1,j) = probability;
    end
    %take the highest value
    [~,minValIndex]=max(probabilities);
    classification(i,1) = classes(minValIndex);
end

%if data set contains the ground truth comput and return the error rate
if(size(dataSet,2)==size(trainingSet,2))
    errorRate=sum(classification(:,1)~=dataSet(:,end))/numRows;
else
    errorRate=-1;
end
end