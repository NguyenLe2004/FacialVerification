import { useEffect, useState } from "react";
import { Button, Col, Form, Row} from "react-bootstrap";
import axios from "axios";
import WebcamComponent from "./WebcamComponent/WebcamComponent";
import "./SignupComponent.css";
const LoginComponent = () => {
  const [validated, setValidated] = useState(false);
  const [isOpenWebcam, setIsOpenWebcam] = useState(false);
  const [encoddedUserImg, setEncodededUserImg] = useState([])
  const [signUpError, setSignUpError] = useState("")
  const [isSignUp, setIsSignUp] = useState(false);
  const [userName, setUserName] = useState("");

  const handleSubmit = async (event) => {
    event.preventDefault();
    event.stopPropagation();

    const form = event.currentTarget;
    if (form.checkValidity()) {
      const data = {
        'user' : form.elements.user.value,
        'encoded_user_img' : encoddedUserImg
      }
      try {
        const response = await axios.post('http://127.0.0.1:4000/api/upload_data', data);
        setUserName(form.elements.user.value);
        if (!response.data.success){
          setSignUpError(response.data[0].message)
        }
        else{
          setIsSignUp(true)
        }
      } catch (error) {
        console.error(error);
        setSignUpError('An error occurred while submitting the form.');
      }
      
    }
    // console.log(error)
    setValidated(true);
  };

  const handleCheckBox = () => {
    setIsOpenWebcam(true);
  };
  return (
    <div>
      {isOpenWebcam &&   <WebcamComponent setEncodededUserImg = {setEncodededUserImg} setIsOpenWebcam = {setIsOpenWebcam} /> }
      {!isSignUp ? (
        <div className={`form-container ${isOpenWebcam ? "hide" : ""}`}>
        <h1 >SIGN UP</h1>
        {signUpError && <div className="error-message">{signUpError}</div>}
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
          <Row>
            <Form.Group className="mb-3">
              <Form.Check
                onClick={handleCheckBox}
                required
                label="Get user image"
                feedback="You must get image before submit"
                checked={encoddedUserImg.length !== 0}
                feedbackType="invalid"
              />
            </Form.Group>
          </Row>
          <Button type="submit">Submit form</Button>
        </Form>
      </div>
      ):(
        <div style={{display:"flex",justifyContent:"center",marginTop:"5vh"}}> 
          <h1>
          Welcome to my website {userName}
          </h1>
        </div>
      )}

    </div>
  );
};

export default LoginComponent;
