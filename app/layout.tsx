import "./global.css";

export const metadata = {
    title: "F1GPT",
    description: " The place to go for all your Formula One questions!"
};

const RootLayout = ({ children }) => {
    return (
        <html lang="en">
            <head>
                <meta charSet="UTF-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1.0" />
                <title>F1Gpt</title>
            </head>
            <body>
                {children}
            </body>
        </html>
    )
};

export default RootLayout;