@echo off
title Push to GitHub - Email Spam Classifier
cd /d "c:\Users\priya\Desktop\email_spam\email-spam-classifier"
echo ===================================================================
echo Pushing Email Spam Classifier code to GitHub...
echo Target Repo: https://github.com/sugap2629-del/Email-spam-classifier.git
echo ===================================================================
echo.
git push -u origin main
echo.
if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] Code successfully pushed to GitHub!
) else (
    echo [NOTE] If you saw an authentication prompt or error,
    echo make sure you sign in to your GitHub account 'sugap2629-del'.
)
echo.
pause
