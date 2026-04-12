import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    Layout.fillWidth: true
    Layout.fillHeight: true

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 12
        spacing: 10

        Label { text: "Export Center"; color: "#E2EEFF"; font.pixelSize: 20; font.bold: true }

        TextArea {
            Layout.fillWidth: true
            Layout.fillHeight: true
            text: appState.selectedStrategyJson()
            readOnly: true
            wrapMode: Text.Wrap
            color: "#D7E5FA"
            background: Rectangle { color: "#0F1725"; radius: 12; border.color: "#1B2A41" }
        }

        RowLayout {
            Layout.fillWidth: true
            Button { text: "Copy Selected Strategy"; onClicked: appState.copySelectedStrategy() }
        }
    }
}
