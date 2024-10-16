USE employees;
-- Question 1
SELECT * FROM employees; 
SELECT * FROM salaries;


-- Question 2
SELECT * FROM salaries WHERE salary*1.7 > 100000;

-- Example 1
SELECT COUNT(DISTINCT last_name) FROM employees;

-- Question 3
SELECT AVG(salary) FROM salaries WHERE emp_no > 1510;

-- Example 2
SELECT last_name, COUNT(last_name) FROM employees GROUP BY last_name;

-- Question 4
SELECT emp_no, AVG(salary) from salaries GROUP BY emp_no;

-- Question 5
SELECT e.first_name, e.last_name, salary FROM employees e JOIN salaries s ON e.emp_no = s.emp_no;

-- Question 6
DELIMITER //
CREATE PROCEDURE emp_avg_salary (IN emp_number INT)
BEGIN
	SELECT AVG(salary) 
    FROM salaries
    WHERE emp_no = emp_number;
END //
DELIMITER ;

CALL emp_avg_salary(11300)



