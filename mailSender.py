import smtplib, ssl

def sendResults(failed, results):
    smtp_server = "smtp.live.com"
    port = 587  # For starttls
    sender_email = "thib_python@outlook.com"
    password = "Python4Ever"
    receiver_email = "thibaud.misonne@student.uclouvain.be"
    if not failed:
        message = f"""
        Subject: Results available

        Hi bro, your code is over ! Here are the results:

        {results}

        NB: Do you know your swag is over 9000 ?

        """
    else:
        message = """
        Subject: ERROR

        Bro, there is an error in your code.

        """

    # Send email here

    # Create a secure SSL context
    context = ssl.create_default_context()

    # Try to log in to server and send email
    try:
        server = smtplib.SMTP(smtp_server, port)
        print("here")
        server.ehlo()  # Can be omitted
        server.starttls(context=context)  # Secure the connection
        server.ehlo()  # Can be omitted
        print("ici")
        server.login(sender_email, password)
        print("la")

        server.sendmail(sender_email, receiver_email, message)

    except Exception as e:
        # Print any error messages to stdout
        print(e)
    finally:
        server.quit()