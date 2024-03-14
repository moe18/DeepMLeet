import streamlit as st
from streamlit_ace import st_ace
from pistonpy import PistonApp

from app.problems import problems

# Instantiate the piston client
piston = PistonApp()

def execute_code(user_code):
    # Execute the user code using pistonpy
    result = piston.run(language="python", version="3.10.0", code=user_code)
    return result


def run_test_cases(user_code, test_cases):
    results = []
    for test_case in test_cases:
        # Modify user_code to include the test case at the end
        code_to_run = f"{user_code}\n\nprint({test_case['test']})"
        result = execute_code(code_to_run)
        
        stdout = result['run']['stdout'].strip()
        expected_output = test_case['expected_output'].strip()
        
        # Check if the test case passed
        passed = stdout == expected_output
        results.append((test_case['test'], expected_output, stdout, passed))
        
    return results

def main():
    st.title("Machine Learning Challenge Platform")
    # Let user select a problem
    problem_names = list(problems.keys())
    selected_problem = st.selectbox("Select a Problem:", problem_names)
    problem_info = problems[selected_problem]

    # Display the selected problem
    st.header(selected_problem)
    st.write(problem_info["description"])

    # Streamlit-ace editor for user code input
    
    user_code = st_ace(language="python", theme="twilight", key=f"code_editor_{selected_problem}", value=problem_info["starter_code"], height=350)
    st.warning('Make sure to apply changes to your code before running it')

    # Button for "Run Code"
    if st.button("Run Code"):
        # Execute user code without testing against test cases
        result = execute_code(user_code)
        if result:
            st.subheader("Execution Result:")
            stdout = result['run']['stdout']
            st.text_area("Output", stdout, height=150)
        else:
            st.error("Failed to execute the code.")

    # Button for "Submit Code"
    if st.button("Submit Code"):
        # Execute user code and run test cases
        results = run_test_cases(user_code, problem_info["test_cases"])
        
        st.subheader("Test Results:")
        for test, expected, output, passed in results:
            if passed:
                st.success(f"Passed: {test} => Expected: {expected}, Got: {output}")
                st.balloons()
            else:
                st.error(f"Failed: {test} => Expected: {expected}, Got: {output}")

    # Button for "Show Solution"
    if st.button("Show Solution"):
        # Display the solution code
        st.subheader("Solution:")
        st.code(problem_info["solution"], language="python")

if __name__ == "__main__":
    main()
