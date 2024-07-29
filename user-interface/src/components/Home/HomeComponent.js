import React from 'react'

import { Routes, Route } from 'react-router-dom';
import SigninComponent from '../Signin/SigninComponent';
import SignupComponent from '../Signup/SignupComponent'
import Header from './Header/Header';
import Body from './Body/Body';
const HomeComponent = () => {
  return (

    <div>
      <Header/>
      <Routes>
        <Route path='/' element = {<Body />} />
        <Route path='/signin' element = {<SigninComponent/> } />
        <Route path='/signup' element = {<SignupComponent/> } />
      </Routes>
    </div>
  )
}

export default HomeComponent;