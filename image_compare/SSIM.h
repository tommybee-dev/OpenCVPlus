//=============================================================================
// Copyright 2019 (c), Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//================================================================================

    #if _MSC_VER > 1000
    #pragma once
    #endif // _MSC_VER > 1000

    #include "TestReport.h"
    #include <opencv2/opencv.hpp>

    #if defined(WIN32) || defined(_WIN64)
    #define CMP_API __cdecl
    #else
    #define CMP_API
    #endif

    using namespace std;
    using namespace cv;

    // CMP_Feedback_Proc
    // Feedback function for conversion.a
    // \param[in] fProgress The percentage progress of the texture compression.
    // \param[in] mipProgress The current MIP level been processed, value of fProgress = mipProgress
    // \return non-NULL(true) value to abort conversion
    typedef bool(CMP_API* CMP_Feedback_Proc)(float fProgress, size_t pUser1, size_t pUser2);

    void getMSE_PSNR( const Mat& I1, const Mat& I2, double  &mse, double &psnr);
    Scalar getSSIM( const Mat& i1, const Mat& i2, CMP_Feedback_Proc pFeedbackProc);
