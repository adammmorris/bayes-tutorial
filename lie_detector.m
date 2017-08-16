%% Task 1 solution
% Adam Morris ? Computational Social Cognition Bootcamp, July 2017


% Lie detector features
probBeepGivenTrue = .2; % false positive rate is 20%
probBeepGivenLie = .7; % false negative rate is 10%

% Returns joint prob(hypothesis, beeps) - i.e. the numerator of Bayes rule.
% To compute this, we need to know (a) the prior of the hypothesis under
%   consideration, (b) a vector of test results (where 1 means detector beeped and 0 means it didn't),
%   and (c) the probability that the detector would beep under the hypothesis.
% Note that "binopdf(X, N, P)" gives you the probability of observing X
%   beeps out of N, where P is the probability of a beep. (It's the probability density function of a
%   binomial distribution.) This is our likelihood function.
getJoint = @(prior, testResults, probBeepGivenHypothesis) ...
    prior * binopdf(sum(testResults), length(testResults), probBeepGivenHypothesis);

% Returns prob(guilty | beeps) - i.e. the posterior probability that the
% person is guilty.
getPosterior = @(priorGuilty, testResults) ...
    getJoint(priorGuilty, testResults, probBeepGivenLie) / ... % numerator of Bayes rule
    (getJoint(priorGuilty, testResults, probBeepGivenLie) + getJoint(1 - priorGuilty, testResults, probBeepGivenTrue)); % denominator

%% Part A

priorInnocent = .2;
priorSuspicious = .8;
testResults = [0]; % it didn't beep

getPosterior(priorInnocent, testResults) % .086
getPosterior(priorSuspicious, testResults) % .60

%% Part B

priorInnocent = .2;
priorSuspicious = .8;
numTests = 11;
posteriors = zeros(numTests, 2);

% Starting with on test results, add successive evidence and compute
% posterior under different priors.
for numNoBeeps = 1:numTests
    testResults = zeros(numNoBeeps - 1, 1);
    posteriors(numNoBeeps, 1) = getPosterior(priorInnocent, testResults);
    posteriors(numNoBeeps, 2) = getPosterior(priorSuspicious, testResults);
end

% Plot results.
plot(0:(numTests - 1), posteriors, 'LineWidth', 4);
xlabel('Number of test results with no beep');
ylabel('Probability of guilt');
set(gca, 'LineWidth', 2);
set(gca, 'FontSize', 36);
legend('Innocent prior', 'Suspicious prior');

%% Part C

% If machine gives fewer false negatives..
probBeepGivenLie = .99;

% Recompute posterior function
getPosterior = @(priorGuilty, testResults) ...
    getJoint(priorGuilty, testResults, probBeepGivenLie) / ... % numerator of Bayes rule
    (getJoint(priorGuilty, testResults, probBeepGivenLie) + getJoint(1 - priorGuilty, testResults, probBeepGivenTrue)); % denominator

% Redo Part B
posteriors = zeros(numTests, 2);
for numNoBeeps = 1:numTests
    testResults = zeros(numNoBeeps - 1, 1);
    posteriors(numNoBeeps, 1) = getPosterior(priorInnocent, testResults);
    posteriors(numNoBeeps, 2) = getPosterior(priorSuspicious, testResults);
end

% Plot results
figure
plot(0:(numTests - 1), posteriors, 'LineWidth', 4);
title('False negative = .01');
xlabel('Number of test results with no beep');
ylabel('Probability of guilt');
set(gca, 'LineWidth', 2);
set(gca, 'FontSize', 36);
legend('Innocent prior', 'Suspicious prior');