import { useState } from "react";
import { Button, Col, Form, Row } from "react-bootstrap";
import WebcamComponent from "./WebcamComponent/WebcamComponent";
import axios from "axios";
import "./SigninComponent.css"

const SigninComponent = () => {
  const [validated, setValidated] = useState(false);
  const [userEncode, setUserEncode] = useState([]);
  const [isSignIn, setIsSignIn] = useState(false);
  const [userName, setUserName] = useState("");
  const [error, setError] = useState(null);

  const handleSubmit = async (event) => {
    event.preventDefault();
    event.stopPropagation();

    const form = event.currentTarget;
    if (form.checkValidity()) {
      const data = {
        user: form.elements.user.value
      };

      try {
        const response = await axios.post("http://127.0.0.1:4000/api/get_user_encode", data, {
          headers: {
            'Content-Type': 'application/json'
          }
        });
        setUserName(form.elements.user.value)
        setUserEncode(response.data);
        setError(null);
      } catch (error) {
        console.error(error);
        setError("We couldn't find your user account. Please check your login credentials or sign up");
      }
    }

    setValidated(true);
  };

  return (
    <div>
      {userEncode.length !== 0 && <WebcamComponent userEncode = {userEncode} setUserEncode = {setUserEncode} setIsSignIn={setIsSignIn}  /> }
      {!isSignIn? (
        <div className="form-container">
        <h1>SIGN IN</h1>
        <Form noValidate validated={validated} onSubmit={handleSubmit}>
          <Row className="mb-3">
            <Form.Group as={Col} controlId="user">
              <Form.Label>User</Form.Label>
              <Form.Control type="text" placeholder="User" required />
              <Form.Control.Feedback type="invalid">
                Please provide a valid User name
              </Form.Control.Feedback>
            </Form.Group>
          </Row>
          <Button type="submit">Submit form</Button>
        </Form>
        {error && <div className="error-message">{error}</div>}
      </div>
      ):(
        <div style={{display:"flex",justifyContent:"center",marginTop:"5vh"}}> 
          <h1>
          Welcome to my website {userName}
          </h1>
        </div>
      ) }

    </div>
  );
}

export default SigninComponent;